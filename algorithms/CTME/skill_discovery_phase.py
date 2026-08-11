"""
Phase 1 of CTME approach: per-agent skill discovery.
"""

import hydra
from omegaconf import OmegaConf
import wandb
import pickle
import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Sequence, NamedTuple, Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

import socialjax
from socialjax.wrappers.baselines import LogWrapper
from utils import batchify, batchify_numpy, unbatchify
from utils import save_params, load_params

from ctme_networks import SkillDiscriminator, SkillActor, SkillCritic

class Transition(NamedTuple):
    obs: jnp.ndarray                                   # (NUM_ACTORS, H, W, C) when in Transition object
    skills_onehot: jnp.ndarray                         # (NUM_ACTORS, N_Z_INDI)
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    log_prob: jnp.ndarray                        
    done: jnp.ndarray
    info: jnp.ndarray


# ============================================================================
# Training Loop
# ============================================================================

def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    n_agents = env.num_agents

    obs_shape = env.observation_space()[0].shape

    NUM_AGENTS = n_agents
    NUM_ENVS = config["NUM_ENVS"]
    NUM_ACTORS = NUM_AGENTS * NUM_ENVS
    NUM_STEPS = config["NUM_STEPS"]
    NUM_UPDATES = int(config["TOTAL_TIMESTEPS"]) // NUM_STEPS // NUM_ENVS
    NUM_EVALS = config["NUM_EVALS"]
    N_Z = config["N_Z"]
    SKILL_INTERVAL = config["SKILL_INTERVAL"]

    assert (
        NUM_STEPS * NUM_ACTORS
    ) % config["L_NUM_MINIBATCHES"] == 0, "NUM_STEPS * NUM_ACTORS must be divisible by L_NUM_MINIBATCHES"
    assert (
        NUM_STEPS * NUM_ACTORS
    ) % config["D_NUM_MINIBATCHES"] == 0, "NUM_STEPS * NUM_ACTORS must be divisible by D_NUM_MINIBATCHES"

    env = LogWrapper(env)

    # NETWORKS
    skill_actor = SkillActor(action_dim=env.action_space().n)
    skill_critic = SkillCritic()
    skill_discriminator = SkillDiscriminator(n_z=N_Z)

    # dummy inputs for networks init
    obs_x = jnp.zeros((1, *obs_shape))
    skill_x = jnp.zeros((1, N_Z))

    def _make_train_state(rng):
        """
        Helper method that inits networks, creates TrainState objects. Called in init_runner_state().
        """
        rng, rng_1, rng_2, rng_3 = jax.random.split(rng, 4)
        skill_actor_params = skill_actor.init(rng_1, obs_x, skill_x)
        skill_critic_params = skill_critic.init(rng_2, obs_x, skill_x)
        skill_discriminator_params = skill_discriminator.init(rng_3, obs_x)

        def _make_tx(lr, max_grad_norm):
            """
            Helper method to create optimizer.
            """
            return optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr, eps=1e-5)
            )

        skill_actor_ts = TrainState.create(
            apply_fn=skill_actor.apply, 
            params=skill_actor_params,
            tx=_make_tx(lr=config["SKILL_ACTOR_LR"], max_grad_norm=config["SKILL_ACTOR_MAX_GRAD_NORM"])
        )
        skill_critic_ts = TrainState.create(
            apply_fn=skill_critic.apply,
            params=skill_critic_params,
            tx=_make_tx(lr=config["SKILL_CRITIC_LR"], max_grad_norm=config["SKILL_CRITIC_MAX_GRAD_NORM"])
        )
        skill_discriminator_ts = TrainState.create(
            apply_fn=skill_discriminator.apply,
            params=skill_discriminator_params,
            tx=_make_tx(lr=config["D_LR"], max_grad_norm=config["D_MAX_GRAD_NORM"])
        )
        train_states = (skill_actor_ts, skill_critic_ts, skill_discriminator_ts)
        return train_states, rng

    def init_runner_state(rng):
        """
        Initialize env, network params, and all runner state.
        """
        train_states, rng = _make_train_state(rng)

        rng, _rng = jax.random.split(rng)
        env_reset_rng = jax.random.split(_rng, NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset)(env_reset_rng)        # obsv: (NUM_ENVS, NUM_AGENTS, H, W, C)

        # sample skills at init
        rng, skill_rng = jax.random.split(rng)
        sampled_skill_ids = jax.random.randint(skill_rng,
            shape=(NUM_ACTORS,),
            minval=0,
            maxval=N_Z
        )
        skills_onehot = jax.nn.one_hot(sampled_skill_ids, num_classes=N_Z)      # (NUM_ACTORS, N_Z)

        # env_step increments every time _env_step() runs
        env_step = 0
        last_done = jnp.zeros((NUM_ACTORS,), dtype=bool)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, env_state, obsv, skills_onehot, last_done, env_step, _rng)
        return runner_state
    
    def _update_step(update_runner_state, unused):
        runner_state, update_steps = update_runner_state

        def _flatten_obs_agent_major(obs):
            """
            Helper method to flatten (NUM_ENVS, NUM_AGENTS, H, W, C) into agent-major NUM_ACTORS.
            """
            return jnp.transpose(obs, (1, 0, 2, 3, 4)).reshape(NUM_ACTORS, *obs_shape)

        def _env_step(runner_state, unused):
            train_states, env_state, obsv, skills_onehot, last_done, env_step, rng = runner_state
            skill_actor_ts, skill_critic_ts, skill_discriminator_ts = train_states

            def _resample_skills(carry):
                rng, _ = carry
                rng, skill_rng = jax.random.split(rng)
                sampled_skill_ids = jax.random.randint(skill_rng,
                    shape=(NUM_ACTORS,),
                    minval=0,
                    maxval=N_Z
                )
                return rng, jax.nn.one_hot(sampled_skill_ids, num_classes=N_Z)      # (NUM_ACTORS, N_Z)
            def _keep_skills(carry):
                rng, skills = carry
                return rng, skills

            # RESAMPLE SKILLS EVERY SKILL_INTERVAL
            should_resample = env_step % SKILL_INTERVAL == 0
            rng, skills_onehot = jax.lax.cond(
                should_resample,
                _resample_skills,
                _keep_skills,
                operand=(rng, skills_onehot)
            )
            
            # flattening to agent-major because rewards/dones/actions are agent-major
                # this is because they use batchify()/batchify_numpy()
            obs_flat = _flatten_obs_agent_major(obsv)       # (NUM_ACTORS, H, W, C)

            # compute critic value
            value = skill_critic.apply(skill_critic_ts.params, obs=obs_flat, skill=skills_onehot)

            # SELECT ACTION
            pi = skill_actor.apply(skill_actor_ts.params, obs=obs_flat, skill=skills_onehot)
            rng, action_rng = jax.random.split(rng)
            # don't need to split key per actor, one operation per key is okay
            actions = pi.sample(seed=action_rng)            # (NUM_ACTORS,)
            log_probs = pi.log_prob(actions)                 # (NUM_ACTORS,)

            # Convert agent-major flat actions back to the env.step format:
                # a list with NUM_AGENTS leaves, each carrying NUM_ENVS actions
            env_actions = unbatchify(actions, env.agents, NUM_ENVS, NUM_AGENTS)
            # make into list with NUM_AGENTS items, each item has actions for all envs (NUM_ENVS,)
            env_actions = [v for v in env_actions.values()]

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_step_rng = jax.random.split(_rng, NUM_ENVS)
            # axes 0, 0, 0 correspond to the 3 inputs and their NUM_ENVS axis
            obsv, env_state, reward, next_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(env_step_rng, env_state, env_actions)

            # CORRECT SHAPES
            reward_flat = batchify_numpy(reward, env.agents, NUM_ACTORS).squeeze()
            next_obs_flat = _flatten_obs_agent_major(obsv)
            next_done_flat = batchify(next_done, env.agents, NUM_ACTORS).squeeze()
            # TODO: double check - transpose info before flatten to make it agent-major
            info = jax.tree_util.tree_map(
                lambda x: jnp.transpose(x, (1, 0)).reshape((NUM_ACTORS,)),
                info
            )

            transition = Transition(
                obs=obs_flat,
                skills_onehot=skills_onehot,
                action=actions,
                value=value,
                reward=reward_flat,
                next_obs=next_obs_flat,
                log_prob=log_probs,
                # this is next_done_flat (t+1) not last_done (t) for the correct GAE mask
                done=next_done_flat,
                info=info
            )
            env_step += 1
            new_runner_state = train_states, env_state, obsv, skills_onehot, next_done_flat, env_step, rng
            return new_runner_state, transition
        
        # ENV STEP LOOP
        runner_state, traj_batch = jax.lax.scan(
            _env_step,
            runner_state, 
            None, 
            NUM_STEPS
        )
        # extract the needed info for update_state and updating process
        train_states, env_state, obsv, skills_onehot, next_done_flat, env_step, rng = runner_state
        
        # UPDATE DISCRIMINATOR
        def _update_discriminator(update_state, unused):
            train_states, traj_batch, rng = update_state
            skill_actor_ts, skill_critic_ts, skill_discriminator_ts = train_states

            def _update_discriminator_minibatch(skill_discriminator_ts, batch):
                next_obs_mb, skills_onehot_mb = batch

                def _discriminator_loss_fn(params):
                    logits = skill_discriminator.apply(params, obs=next_obs_mb)
                    loss = optax.softmax_cross_entropy(logits, skills_onehot_mb).mean()
                    accuracy = (
                        jnp.argmax(logits, axis=-1)
                        == jnp.argmax(skills_onehot_mb, axis=-1)
                    ).astype(jnp.float32).mean()
                    return loss, accuracy

                grad_fn = jax.value_and_grad(_discriminator_loss_fn, has_aux=True)
                (loss, accuracy), grads = grad_fn(skill_discriminator_ts.params)
                skill_discriminator_ts = skill_discriminator_ts.apply_gradients(grads=grads)

                loss_info = {
                    "train/disc_loss": loss,
                    "train/disc_accuracy": accuracy,
                    "train/disc_grad_norm": optax.global_norm(grads),
                }
                return skill_discriminator_ts, loss_info

            rng, _rng = jax.random.split(rng)
            batch_size = NUM_STEPS * NUM_ACTORS
            next_obs_flat = traj_batch.next_obs.reshape(batch_size, *obs_shape)
            skills_onehot_flat = traj_batch.skills_onehot.reshape(batch_size, N_Z)

            permutation = jax.random.permutation(_rng, batch_size)
            batch = (next_obs_flat, skills_onehot_flat)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0),
                batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x,
                    [config["D_NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch
            )

            skill_discriminator_ts, loss_info = jax.lax.scan(
                _update_discriminator_minibatch,
                skill_discriminator_ts,
                minibatches,
            )
            
            new_train_states = (skill_actor_ts, skill_critic_ts, skill_discriminator_ts)
            new_update_state = (new_train_states, traj_batch, rng)
            
            return new_update_state, loss_info
        
        # DISCRIMINATOR UPDATE LOOP
        update_state = (train_states, traj_batch, rng)
        update_state, disc_loss_info = jax.lax.scan(
            _update_discriminator,
            update_state,
            None,
            config["D_UPDATE_EPOCHS"]
        )

        # COMPUTE INTRINSIC REWARDS FROM UPDATED DISCRIMINATOR
        train_states, traj_batch, rng = update_state
        skill_actor_ts, skill_critic_ts, skill_discriminator_ts = train_states

        next_obs_flat = traj_batch.next_obs.reshape(NUM_STEPS * NUM_ACTORS, *obs_shape)
        skills_onehot_flat = traj_batch.skills_onehot.reshape(NUM_STEPS * NUM_ACTORS, N_Z)

        logits = skill_discriminator.apply(skill_discriminator_ts.params, obs=next_obs_flat)
        log_q = jax.nn.log_softmax(logits, axis=-1)
        log_q_flat = jnp.sum(log_q * skills_onehot_flat, axis=-1)

        # log q(z|o) + log(N_Z) because log p(z) = log(1 / N_Z) = -log(N_Z)
        intrinsic_flat = log_q_flat + jnp.log(N_Z)
        intrinsic = intrinsic_flat.reshape(NUM_STEPS, NUM_ACTORS)
        # note: LAMBDA_ENV set to 0 currently
        combined_reward = config["LAMBDA_ENV"] * traj_batch.reward + config["LAMBDA_SKILL"] * intrinsic

        # CALCULATE GAE
        last_obs_flat = _flatten_obs_agent_major(obsv)          # (NUM_ACTORS, H, W, C)
        last_value = skill_critic.apply(skill_critic_ts.params, obs=last_obs_flat, skill=skills_onehot)

        def _calculate_gae(traj_batch, rewards, last_value):
            def _get_advantages(gae_and_next_value, transition_and_reward):
                gae, next_value = gae_and_next_value
                transition, reward = transition_and_reward
                done, value = transition.done, transition.value
                not_done = 1.0 - done.astype(jnp.float32)

                delta = reward + config["L_GAMMA"] * next_value * not_done - value
                gae = (
                    delta
                    + config["L_GAMMA"] * config["L_GAE_LAMBDA"] * not_done * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_value), last_value),
                (traj_batch, rewards),
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, combined_reward, last_value)
        
        # UPDATE SKILL ACTOR AND CRITIC
        def _update_skill_actor_critic(update_state, unused):
            train_states, traj_batch, advantages, targets, rng = update_state
            skill_actor_ts, skill_critic_ts, skill_discriminator_ts = train_states

            def _update_minibatch(train_states, batch_info):
                skill_actor_ts, skill_critic_ts = train_states
                traj_batch, advantages, targets = batch_info

                def _skill_actor_loss_fn(params, traj_batch, advantages):
                    pi = skill_actor.apply(params, obs=traj_batch.obs, skill=traj_batch.skills_onehot)
                    log_prob = pi.log_prob(traj_batch.action)
                    log_ratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(log_ratio)

                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    loss_actor1 = ratio * advantages
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["L_CLIP_EPS"],
                            1.0 + config["L_CLIP_EPS"],
                        )
                        * advantages
                    )
                    policy_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()
                    actor_loss = policy_loss - config["L_ENT_COEF"] * entropy

                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = (jnp.abs(ratio - 1.0) > config["L_CLIP_EPS"]).astype(jnp.float32).mean()

                    return actor_loss, (policy_loss, entropy, ratio.mean(), approx_kl, clip_frac)

                def _skill_critic_loss_fn(params, traj_batch, targets):
                    value = skill_critic.apply(params, obs=traj_batch.obs, skill=traj_batch.skills_onehot)
                    value_pred_clipped = traj_batch.value + jnp.clip(
                        value - traj_batch.value,
                        -config["L_CLIP_EPS"],
                        config["L_CLIP_EPS"]
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(
                        value_losses,
                        value_losses_clipped,
                    ).mean()
                    critic_loss = config["L_VF_COEF"] * value_loss

                    return critic_loss, (value_loss, value.mean())

                # SKILL ACTOR GRADIENTS
                actor_grad_fn = jax.value_and_grad(_skill_actor_loss_fn, has_aux=True)
                (actor_loss, actor_aux), actor_grads = actor_grad_fn(
                    skill_actor_ts.params,
                    traj_batch,
                    advantages,
                )

                # SKILL CRITIC GRADIENTS
                critic_grad_fn = jax.value_and_grad(_skill_critic_loss_fn, has_aux=True)
                (critic_loss, critic_aux), critic_grads = critic_grad_fn(
                    skill_critic_ts.params,
                    traj_batch,
                    targets,
                )

                skill_actor_ts = skill_actor_ts.apply_gradients(grads=actor_grads)
                skill_critic_ts = skill_critic_ts.apply_gradients(grads=critic_grads)

                policy_loss, entropy, ratio_mean, approx_kl, clip_frac = actor_aux
                value_loss, value_mean = critic_aux

                actor_grad_norm = optax.global_norm(actor_grads)
                critic_grad_norm = optax.global_norm(critic_grads)

                loss_info = {
                    "train/l_total_loss": actor_loss + critic_loss,
                    "train/l_actor_loss": actor_loss,
                    "train/l_policy_loss": policy_loss,
                    "train/l_critic_loss": critic_loss,
                    "train/l_value_loss": value_loss,
                    "train/l_value_mean": value_mean,
                    "train/l_entropy": entropy,
                    "train/l_ratio": ratio_mean,
                    "train/l_approx_kl": approx_kl,
                    "train/l_clip_frac": clip_frac,
                    "train/l_actor_grad_norm": actor_grad_norm,
                    "train/l_critic_grad_norm": critic_grad_norm,
                }
                return (skill_actor_ts, skill_critic_ts), loss_info

            # BUILD MINIBATCHES
            rng, _rng = jax.random.split(rng)
            batch_size = NUM_STEPS * NUM_ACTORS
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]),
                batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0),
                batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x,
                    [config["L_NUM_MINIBATCHES"], -1] + list(x.shape[1:]),
                ),
                shuffled_batch
            )

            (skill_actor_ts, skill_critic_ts), loss_info = jax.lax.scan(
                _update_minibatch,
                (skill_actor_ts, skill_critic_ts),
                minibatches,
            )

            new_train_states = (skill_actor_ts, skill_critic_ts, skill_discriminator_ts)
            new_update_state = (new_train_states, traj_batch, advantages, targets, rng)

            return new_update_state, loss_info
        
        # SKILL ACTOR-CRITIC UPDATE LOOP
        update_state = (train_states, traj_batch, advantages, targets, rng)
        update_state, actor_critic_loss_info = jax.lax.scan(
            _update_skill_actor_critic,
            update_state,
            None,
            config["L_UPDATE_EPOCHS"]
        )

        train_states, traj_batch, advantages, targets, rng = update_state
        update_steps += 1

        def callback(metric):
            metric = jax.device_get(metric)
            wandb.log(metric, step=int(metric["env_step"]))

        # POPULATE METRIC DICT
        def _reduce_info_metric(info):
            """
            Helper to track environment behavior diagnostics. Don't care about 
            return/extrinsic rewards right now (unsupervised skill discovery).

            clean_action_total: TOTAL number of clean actions taken during rollout
            clean_action_mean: average clean action RATE per timestep/actor
            cleaned_water_mean: cleanliness of river on average across rollout
            cleaned_water_final: final state of environment cleanliness
            """
            info_metric = {}

            clean = info["clean_action_info"]
            info_metric["rollout/clean_action_total"] = clean.sum()
            info_metric["rollout/clean_action_mean"] = clean.mean()

            water = info["cleaned_water"]
            info_metric["rollout/cleaned_water_mean"] = water.mean()
            info_metric["rollout/cleaned_water_final"] = water[-1].mean()

            return info_metric

        info_metric = _reduce_info_metric(traj_batch.info)
        train_metric = jax.tree_util.tree_map(lambda x: x.mean(), actor_critic_loss_info)
        disc_metric = jax.tree_util.tree_map(lambda x: x.mean(), disc_loss_info)
        metric = {
            **info_metric,
            **disc_metric,
            **train_metric,
            "train/intrinsic_reward_mean": intrinsic.mean(),
            "train/intrinsic_reward_std": intrinsic.std(),
            "train/combined_reward_mean": combined_reward.mean(),
            "train/advantage_mean": advantages.mean(),
            "train/target_mean": targets.mean(),
            "update_step": update_steps,
            "env_step": update_steps * NUM_STEPS * NUM_ENVS,
        }

        jax.debug.callback(callback, metric)

        runner_state = (train_states, env_state, obsv, skills_onehot, next_done_flat, env_step, rng)
        new_update_runner_state = (runner_state, update_steps)
        return new_update_runner_state, metric
    
    chunk_updates = max(1, NUM_UPDATES // NUM_EVALS)
    remainder_updates = NUM_UPDATES - (NUM_EVALS * chunk_updates)

    def train_chunk(update_runner_state):
        return jax.lax.scan(_update_step, update_runner_state, None, chunk_updates)

    if remainder_updates > 0:
        def remainder_chunk(update_runner_state):
            return jax.lax.scan(_update_step, update_runner_state, None, remainder_updates)
    else:
        remainder_chunk = None

    return init_runner_state, train_chunk, remainder_chunk, chunk_updates, remainder_updates
        
def evaluate(params, env, save_path, config, wandb_step: int, log_gif: bool = False):
    actor_params, discriminator_params = params

    rng = jax.random.PRNGKey(0)
    n_z = config["N_Z"]
    obs_shape = env.observation_space()[0].shape

    skill_actor = SkillActor(action_dim=env.action_space().n, activation=config["ACTIVATION"])
    skill_discriminator = SkillDiscriminator(n_z=n_z, activation=config["ACTIVATION"])

    if save_path is not None:
        root_dir = Path(save_path)
    else:
        root_dir = Path("evaluation/ctme_skill_discovery/cleanup")
    root_dir.mkdir(parents=True, exist_ok=True)

    EVAL_NUM_EPISODES = config["EVAL_NUM_EPISODES"]
    NUM_STEPS = config["NUM_STEPS"]
    SKILL_NUM_STEPS = EVAL_NUM_EPISODES * NUM_STEPS
    eval_metrics = {"eval/episodes_averaged": EVAL_NUM_EPISODES}

    for skill_id in range(n_z):
        skill_disc_acc_sum = 0.0
        skill_intrinsic_sum = 0.0
        skill_entropy_sum = 0.0

        skill_clean_action_total_sum = 0.0
        skill_clean_action_mean_sum = 0.0
        skill_cleaned_water_mean_sum = 0.0
        skill_cleaned_water_final_sum = 0.0

        skill_pics = []

        # shape = (NUM_AGENTS, N_Z)
        fixed_skill_onehot = jax.nn.one_hot(jnp.full((env.num_agents,), skill_id), num_classes=n_z)

        for episode_idx in range(EVAL_NUM_EPISODES):
            rng, reset_rng = jax.random.split(rng)
            obs, state = env.reset(reset_rng)
            # episode_pics is temp storage for frames before being copied to skill_pics
            episode_pics = []
            episode_cleaned_water_values = []

            # only save 1 GIF per skill (1st episode / 10)
            if log_gif and episode_idx == 0:
                # append initial frame
                episode_pics.append(env.render(state))

            for _ in range(NUM_STEPS):
                obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
                    env.num_agents,
                    *obs_shape,
                )

                # GET ACTIONS
                pi = skill_actor.apply(
                    actor_params,
                    obs=obs_batch,
                    skill=fixed_skill_onehot,
                )
                # deterministic eval
                actions = pi.mode()
                entropy = pi.entropy().mean()

                env_act = {
                    k: v.squeeze()
                    for k, v in unbatchify(actions, env.agents, 1, env.num_agents).items()
                }

                rng, step_rng = jax.random.split(rng)
                obs, state, reward, done, info = env.step(
                    step_rng,
                    state,
                    [v.item() for v in env_act.values()],
                )

                # DISCRIMINATOR PREDICTIONS
                next_obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
                    env.num_agents,
                    *obs_shape,
                )
                logits = skill_discriminator.apply(
                    discriminator_params,
                    obs=next_obs_batch,
                )
                pred_skill = jnp.argmax(logits, axis=-1)
                disc_acc = (pred_skill == skill_id).astype(jnp.float32).mean()

                log_q = jax.nn.log_softmax(logits, axis=-1)[:, skill_id]
                intrinsic = log_q + jnp.log(n_z)

                skill_disc_acc_sum += float(disc_acc)
                skill_intrinsic_sum += float(intrinsic.mean())
                skill_entropy_sum += float(entropy)

                # TRACK CLEAN METRICS (ACTION / WATER)
                clean_action = jnp.asarray(info["clean_action_info"])
                skill_clean_action_total_sum += float(clean_action.sum())
                skill_clean_action_mean_sum += float(clean_action.mean())

                cleaned_water = jnp.asarray(info["cleaned_water"])
                episode_cleaned_water_values.append(float(cleaned_water.mean()))

                # only save frames for first episode
                if log_gif and episode_idx == 0:
                    episode_pics.append(env.render(state))

            skill_cleaned_water_mean_sum += float(np.mean(episode_cleaned_water_values))
            skill_cleaned_water_final_sum += float(episode_cleaned_water_values[-1])

            if log_gif and episode_idx == 0:
                skill_pics = episode_pics

        # skill_step_count = EVAL_NUM_EPISODES * NUM_STEPS = 10,000

        prefix = f"eval/skill_{skill_id}"

        eval_metrics[f"{prefix}/disc_accuracy"] = skill_disc_acc_sum / SKILL_NUM_STEPS
        eval_metrics[f"{prefix}/intrinsic_reward_mean"] = skill_intrinsic_sum / SKILL_NUM_STEPS
        eval_metrics[f"{prefix}/action_entropy"] = skill_entropy_sum / SKILL_NUM_STEPS

        eval_metrics[f"{prefix}/clean_action_total"] = skill_clean_action_total_sum / EVAL_NUM_EPISODES
        eval_metrics[f"{prefix}/clean_action_mean"] = skill_clean_action_mean_sum / SKILL_NUM_STEPS
        eval_metrics[f"{prefix}/cleaned_water_mean"] = skill_cleaned_water_mean_sum / EVAL_NUM_EPISODES
        eval_metrics[f"{prefix}/cleaned_water_final"] = skill_cleaned_water_final_sum / EVAL_NUM_EPISODES

        if log_gif and skill_pics:
            new_pics = [Image.fromarray(np.array(img)) for img in skill_pics]
            gif_path = root_dir / (
                f"skill-{skill_id}_seed-{config['SEED']}_step-{int(wandb_step)}_frames-{len(new_pics)}.gif"
            )
            new_pics[0].save(
                gif_path,
                format="GIF",
                save_all=True,
                optimize=False,
                append_images=new_pics[1:],
                duration=200,
                loop=0,
            )
            wandb.log(
                {
                    f"{prefix}/episode_gif": wandb.Video(
                        str(gif_path),
                        caption=f"Forced Skill {skill_id}",
                        format="gif",
                    )
                },
                step=int(wandb_step),
            )

    wandb.log(eval_metrics, step=int(wandb_step))


def single_run(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["CTME", "cleanup"],
        group=config["WANDB_GROUP"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"ctme_skill_disc_cleanup",
    )

    init_runner_state, train_chunk, remainder_chunk, chunk_updates, remainder_updates = make_train(config)
    init_jit = jax.jit(init_runner_state)

    # chunk_jit is a scan for _update_step() over chunk_updates == 3906 // 10 = 390 updates
    chunk_jit = jax.jit(train_chunk)

    remainder_jit = None
    if remainder_chunk is not None:
        remainder_jit = jax.jit(remainder_chunk)   

    NUM_EVALS = config["NUM_EVALS"]

    rng = jax.random.PRNGKey(config["SEED"])
    print(f"config seed: {config['SEED']}")
    runner_state = init_jit(rng)
    # update_runner_state holds runner_state and update_steps
    update_runner_state = (runner_state, 0)

    for i in range(NUM_EVALS):
        # runner_state: (runner_state, update_steps)
        # _ is metric from update_step() return
        update_runner_state, _ = chunk_jit(update_runner_state)

        # extract params for eval
        skill_actor_ts, skill_critic_ts, skill_discriminator_ts = update_runner_state[0][0]
        actor_params = skill_actor_ts.params
        discriminator_params = skill_discriminator_ts.params

        update_step = int(update_runner_state[1])
        env_step = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

        # only log (5 * N_Z) + 1 (bc of last chunk) GIFs
        log_gif = i % 2 == 0

        # setting wandb_step so it's more clear with eval and train metrics
        # wandb_step=env_step, not update_step -> logging is comparable if NUM_STEPS or NUM_ENVS changes
        evaluate(
            (actor_params, discriminator_params),
            socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
            save_path=None,
            config=config,
            wandb_step=env_step,
            log_gif=log_gif
        )

    # REMAINDER CHUNK
    # Since NUM_UPDATES not divisible by 10, do remainder updates chunk and 1 final eval for 11 evals total
    if remainder_updates > 0:
        update_runner_state, _ = remainder_jit(update_runner_state)

    # final eval after all updates
    skill_actor_ts, skill_critic_ts, skill_discriminator_ts = update_runner_state[0][0]
    # actor params
    actor_params = skill_actor_ts.params
    discriminator_params = skill_discriminator_ts.params
    
    update_step = int(update_runner_state[1])
    env_step = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

    evaluate(
        (actor_params, discriminator_params),
        socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
        save_path=None,
        config=config,
        wandb_step=env_step,
        log_gif=True,
    )
    print("Finished training and evals")
    
@hydra.main(version_base=None, config_path="config", config_name="ctme_skill_discovery_cleanup_mini")
def main(config):
    print("Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("Starting skill learning phase.")

    single_run(config)
    

if __name__ == "__main__":
    main()
