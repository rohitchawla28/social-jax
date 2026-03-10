""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
Fixed bugs of jaxmarl
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import MAPPOWorldStateWrapper, LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        # x = nn.Conv(
        #     features=32,
        #     kernel_size=(3, 3),
        #     kernel_init=orthogonal(np.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(x)
        # x = activation(x)
        # x = nn.Conv(
        #     features=32,
        #     kernel_size=(3, 3),
        #     kernel_init=orthogonal(np.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(x)
        # x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class Actor(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        embedding = CNN(self.activation)(obs)
        actor_mean = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)

        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi


class Critic(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        world_state = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(world_state)
        hidden = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)

        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)
        # squeeze 去除最后一个维度
        return jnp.squeeze(value, axis=-1)
    

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_numpy(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = MAPPOWorldStateWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # INIT NETWORK
    actor_network = Actor(env.action_space().n, activation=config["ACTIVATION"])
    critic_network = Critic(activation=config["ACTIVATION"])

    ac_init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

    obs_shape = env.observation_space()[0].shape
    global_shape = (1, *obs_shape[:-1], obs_shape[-1] * env.num_agents)
    cr_init_x = jnp.zeros(global_shape)
    # cr_init_x = jnp.zeros((1, *(env.observation_space()[0]).shape)) 

    # helper method for init_runner_state() to create TrainState object
    def _make_train_state(rng):
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        actor_network_params = actor_network.init(_rng_actor, ac_init_x)
        critic_network_params = critic_network.init(_rng_critic, cr_init_x)

        if config["ANNEAL_LR"]:            
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        # NOTE: they had this incorrectly use actor_network.apply before
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )
        return (actor_train_state, critic_train_state), rng
    
    # new method for chunk refactor
    # runner state init method called once only
    def init_runner_state(rng):
        train_states, rng = _make_train_state(rng)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        rng, _rng = jax.random.split(rng)

        # not sure why but MAPPO_cleanup has "done" in the runner_state object
            # update_steps counter is in the update_runner_state object
        # on other hand, IPPO has update_steps counter in runner_state object w/o "done"
        return (train_states, env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), _rng)

    # TRAIN LOOP
    def _update_step(update_runner_state, unused):
        # COLLECT TRAJECTORIES
        runner_state, update_steps = update_runner_state
        def _env_step(runner_state, unused):
            train_states, env_state, last_obs, last_done, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)


            obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
            # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

            # print("input_obs_shape", obs_batch.shape)

            pi = actor_network.apply(train_states[0].params, obs_batch)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(
                action, env.agents, config["NUM_ENVS"], env.num_agents
            )

            # env_act = {k: v.flatten() for k, v in env_act.items()}
            env_act = [v for v in env_act.values()]

            #VALUE
            world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1)
            world_state = jnp.expand_dims(world_state, axis=0)
            world_state = jnp.tile(world_state, (env.num_agents, 1, 1, 1, 1))
            world_state = jnp.reshape(world_state, (-1, *(world_state.shape[2:])))
            # world_state = last_obs["world_state"].swapaxes(0,1)  
            # world_state = world_state.reshape((config["NUM_ACTORS"],-1))
            value = critic_network.apply(train_states[1].params, world_state)
            # expanded_value = jnp.expand_dims(value, axis=0)
            # value = jnp.tile(expanded_value, (env.num_agents, 1))
            value = value.reshape(config["NUM_ACTORS"])


            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])

            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)

            # INCORRECT prev code: info from env.step is env-major, doesn't align w/ obs_batch actor axis (agent-major)
                # info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

            # transpose info before flatten to make it agent-major
            info = jax.tree_util.tree_map(
                lambda x: jnp.transpose(x, (1, 0)).reshape((config["NUM_ACTORS"],)),
                info,
            )

            done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
            transition = Transition(
                jnp.tile(done["__all__"], env.num_agents),
                last_done, # TODO: i think this should be done_batch instead of last_done
                action,
                value,
                batchify_numpy(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                log_prob,
                obs_batch,
                world_state,
                info
            )
            runner_state = (train_states, env_state, obsv, done_batch, rng)
            return runner_state, transition

        # NOTE: _env_step() will ONLY run 100 times bc config
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        train_states, env_state, last_obs, last_done, rng = runner_state

        # last_world_state = last_obs["world_state"].swapaxes(0,1)
        # last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
        last_world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1)
        last_val = critic_network.apply(train_states[1].params, last_world_state)
        last_val = jnp.expand_dims(last_val, axis=0)
        last_val = jnp.tile(last_val, (env.num_agents, 1))
        last_val = last_val.reshape((config["NUM_ACTORS"],-1))
        last_val = last_val.squeeze()

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_states, batch_info):
                actor_train_state, critic_train_state = train_states
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(actor_params, traj_batch, gae):
                    # RERUN NETWORK
                    pi = actor_network.apply(
                        actor_params,
                        jnp.reshape(traj_batch.obs, (-1, *(traj_batch.obs).shape[-3:])),
                    ) #.reshape(traj_batch.action.shape)

                    log_prob = pi.log_prob(traj_batch.action.reshape(-1))

                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - jnp.reshape(traj_batch.log_prob, (-1,))
                    logratio = jnp.reshape(logratio, traj_batch.action.shape)
                    log_prob = jnp.reshape(log_prob, traj_batch.action.shape)
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()
                    # debug
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                    
                    actor_loss = (
                        loss_actor
                        - config["ENT_COEF"] * entropy
                    )
                    return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                def _critic_loss_fn(critic_params, traj_batch, targets):
                    # RERUN NETWORK
                    # they don't use apply_fn from TrainState object so maybe not an issue that they had actor_network
                    value = critic_network.apply(critic_params, traj_batch.world_state.reshape(-1, *(traj_batch.world_state).shape[-3:])) 
                    value = jnp.reshape(value, traj_batch.value.shape)
                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )
                    critic_loss = config["VF_COEF"] * value_loss

                    # TODO: this is same spot for value_mean i did in IPPO, 
                        # but should it come from env_step value? what does the env_step value do?
                    # log critic prediction value as well for analyzing training dynamics
                    value_mean = value.mean()

                    return critic_loss, (value_loss, value_mean)

                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss, actor_grads = actor_grad_fn(
                    actor_train_state.params, traj_batch, advantages
                )

                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                # critic_grad_fn will return this form: ((value, auxiliary_data), gradient)
                # i think critic_loss (HERE) is a tuple: (critic_loss, (value_loss, value_mean))
                critic_loss, critic_grads = critic_grad_fn(
                    critic_train_state.params, traj_batch, targets
                )
                    
                actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                
                total_loss = actor_loss[0] + critic_loss[0]

                # calculate gradient norms for additional training metric
                actor_grad_norm = optax.global_norm(actor_grads)
                critic_grad_norm = optax.global_norm(critic_grads)

                # TODO: figure out is it significance of value_loss * config["VF_COEF"] to give critic loss (is it same as value_loss?)
                loss_info = {
                    "train/total_loss": total_loss,
                    "train/actor_loss": actor_loss[0],
                    "train/value_loss": critic_loss[0],
                    "train/value_mean": critic_loss[1][1],
                    "train/actor_grad_norm": actor_grad_norm,
                    "train/critic_grad_norm": critic_grad_norm,
                    "train/entropy": actor_loss[1][1],
                    "train/ratio": actor_loss[1][2],
                    "train/approx_kl": actor_loss[1][3],
                    "train/clip_frac": actor_loss[1][4],
                }
                    
                return (actor_train_state, critic_train_state), loss_info

            (train_states, traj_batch, advantages, targets, rng) = update_state

            rng, _rng = jax.random.split(rng)

            # batch = (
            #     traj_batch,
            #     advantages.squeeze(),
            #     targets.squeeze(),
            # )
            # permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

            # shuffled_batch = jax.tree_util.tree_map(
            #     lambda x: jnp.take(x, permutation, axis=1), batch
            # )
            
            # minibatches = jax.tree_util.tree_map(
            #     lambda x: jnp.swapaxes(
            #         jnp.reshape(
            #             x,
            #             [x.shape[0], config["NUM_MINIBATCHES"], -1]
            #             + list(x.shape[2:]),
            #         ),
            #         1,
            #         0,
            #     ),
            #     shuffled_batch,
            # )
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
            ), "batch size must be equal to number of steps * number of actors"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            train_states, loss_info = jax.lax.scan(
                _update_minbatch, train_states, minibatches
            )
            update_state = (
                train_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, loss_info

        update_state = (train_states, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_states = update_state[0]
        metric = traj_batch.info
        # loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
        # loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
        # metric["loss"] = loss_info

        # avg the training metrics over epochs and minibatches to get 1 scalar per metric each update_step
        # this matches their logging granularity
        train_metric = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
        metric = {**metric, **train_metric}

        rng = update_state[-1]

        def callback(metric):
            # filter out NaNs bc didn't want to log 0s earlier
            episodic_keys = ("returned_episode_returns", "returned_episode_lengths", "returned_episode")
            filtered_metric = {}
            for k, v in metric.items():
                if k in episodic_keys and np.isnan(np.asarray(v)).any():
                    continue
                filtered_metric[k] = v
            wandb.log(filtered_metric, step=metric["env_step"])

        update_steps = update_steps + 1

            # reduce per-update metrics correctly. some metrics are episode level and only become non-zero at end of episode
        def _reduce_metric_dict(m):
            out = {}

            episode_keys = ("returned_episode_returns", "returned_episode_lengths", "returned_episode")
            episode_mask = (m["returned_episode"] > 0).astype(jnp.float32)
            denom = episode_mask.sum()

            # added this masking because don't want to log zeros, should just log when an episode finishes (after 10 rollouts)
            def episodic_mean(val):
                num = (val * episode_mask).sum()
                return jnp.where(denom > 0, num / denom, jnp.nan)

            if "raw_reward_individual" in m:
                raw = m["raw_reward_individual"]  # shape=(T, NUM_ACTORS) after flatten

                # unflatten to (T, num_agents, NUM_ENVS) assuming agent-major flattening
                if raw.ndim == 2:
                    raw = raw.reshape((raw.shape[0], env.num_agents, config["NUM_ENVS"]))

                # episode/chunk return per agent per env
                ep_returns_agent_env = raw.sum(axis=0)  # (num_agents, NUM_ENVS)

                # avg across envs
                mean_ep_return_per_agent = ep_returns_agent_env.mean(axis=1)  # (num_agents,)

                # sum across agents to get team return, then avg across envs
                mean_ep_return_team = ep_returns_agent_env.sum(axis=0).mean()  # scalar

                # fairness (variance) between agents, averaged across envs
                mean_ep_return_variance = ep_returns_agent_env.var(axis=0).mean()  # scalar

                # fairness (pairwise absolute difference)
                # TODO: i think this doesn't work bc some jax boolean error can leave it out for now
                # diff = jnp.abs(ep_returns_agent_env[:, None, :] - ep_returns_agent_env[None, :, :])  # (A, A, E)
                # pair_mask = jnp.triu(jnp.ones((env.num_agents, env.num_agents), dtype=bool), k=1)
                # mean_ep_pairwise_absdiff = diff[pair_mask].mean()

                for i in range(env.num_agents):
                    out[f"rollout/raw_ep_return_agent{i}"] = mean_ep_return_per_agent[i]

                out["rollout/raw_ep_return_team"] = mean_ep_return_team
                out["rollout/raw_ep_return_variance"] = mean_ep_return_variance
                # out["rollout/raw_ep_pairwise_absdiff"] = mean_ep_pairwise_absdiff

            for k, v in m.items():
                if k == "raw_reward_individual":
                    # don't want to log per-step raw rewards directly
                    continue
                if k in episode_keys:
                    out[k] = episodic_mean(v)
                elif k == "clean_action_info":
                    # per-step action so sum over the rollout
                    out["rollout/clean_action_chunk"] = v.sum()
                elif k == "cleaned_water":
                    # state metric so both mean and final are useful
                    out["rollout/cleaned_water_mean"] = v.mean()
                    out["rollout/cleaned_water_final"] = v[-1].mean()
                else:
                    out[k] = v.mean()

            return out
            
        metric = _reduce_metric_dict(metric)
        
        metric["update_step"] = update_steps
        metric["env_step"] = update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]
        # metric["clean_action_info"] = metric["clean_action_info"] * config["ENV_KWARGS"]["num_inner_steps"]

        # jax.experimental.io_callback(callback, None, metric)

        jax.debug.callback(callback, metric)
        
        runner_state = (train_states, env_state, last_obs, last_done, rng)
        # NOTE: update_steps is in tuple, different from ippo structure
        return (runner_state, update_steps), metric
    
    num_evals = 10
    chunk_updates = max(1, config["NUM_UPDATES"] // num_evals)
    remainder_updates = config["NUM_UPDATES"] - (num_evals * chunk_updates)

    def train_chunk(update_runner_state):
        return jax.lax.scan(_update_step, update_runner_state, None, chunk_updates)
    
    if remainder_updates > 0:
        def remainder_chunk(update_runner_state):
            return jax.lax.scan(_update_step, update_runner_state, None, remainder_updates)
    else:
        remainder_chunk = None

    return init_runner_state, train_chunk, remainder_chunk, chunk_updates, remainder_updates

def single_run(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAPPO", "cleanup"],
        group=config["WANDB_GROUP"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'mappo_cnn_cleanup'
    )

    # BEFORE REFACTOR
        # rng = jax.random.PRNGKey(config["SEED"])
        # rngs = jax.random.split(rng, config["NUM_SEEDS"])
        # # train_jit = jax.jit(make_train(config))

        # train_jit = make_train(config)

        # out = jax.vmap(train_jit)(rngs)

    # ===== CHUNKING REFACTOR ======
    # NOTE: they didn't have it all in a jit(), don't know why, but i added it
    init_runner_state, train_chunk, remainder_chunk, chunk_updates, remainder_updates = make_train(config)
    init_jit = jax.jit(init_runner_state)

    # chunk_jit is a scan for _update_step() over chunk_updates == 3906 // 10 = 390 updates
    chunk_jit = jax.jit(train_chunk)

    remainder_jit = None
    if remainder_chunk is not None:
        remainder_jit = jax.jit(remainder_chunk)
    
    rng = jax.random.PRNGKey(config["SEED"])

    print(f"config seed: {config['SEED']}")
    runner_state = init_jit(rng)
    update_runner_state = (runner_state, 0)

    num_evals = 10
    for k in range(num_evals):
        # runner_state: (runner_state, update_steps)
        # _ is metric from update_step() return
        update_runner_state, _ = chunk_jit(update_runner_state)

        # extract params for eval. runner_state[0] gives runner_state. [0] again gives train_states
        train_states = update_runner_state[0][0]
        # get actor params
        params = train_states[0].params

        update_step = int(update_runner_state[1])
        env_step = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

        # log eval metrics exactly 10 times
        # setting wandb_step so it's more clear with eval and train metrics
            # setting it to env_step instead of update_step so that if i change NUM_STEPS or NUM_ENVS, logging still comparable
        evaluate(
            params,
            socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
            save_path=None,
            config=config,
            wandb_step=env_step,
            log_gif=False
        )
    
    # REMAINDER CHUNK
    # Since NUM_UPDATES not divisible by 10, do remainder updates chunk and 1 final eval for 11 evals total
    if remainder_updates > 0:
        update_runner_state, _ = remainder_jit(update_runner_state)

    # final eval after all updates, only eval that logs GIF
    train_states = update_runner_state[0][0]
    # actor params
    params = train_states[0].params
    
    update_step = int(update_runner_state[1])
    env_step = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

    evaluate(
        params,
        socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
        save_path=None,
        config=config,
        wandb_step=env_step,
        log_gif=True,
    )
    print("Finished training and evals")

    # their old eval stuff
        # print("** Saving Results **")
        # filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
        # train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0][0][0])
        # save_path = f"./checkpoints/{filename}.pkl"
        # save_params(train_state, save_path)
        # params = load_params(save_path)
        # print("** Evaluating Results **")
        # evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)

    # idk what this is
        # state_seq = get_rollout(train_state.params, config)
        # viz = OvercookedVisualizer()
        # agent_view_size is hardcoded as it determines the padding around the layout.
        # viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)

def evaluate(params, env, save_path, config, wandb_step: int, log_gif: bool = False):
    rng = jax.random.PRNGKey(0)
    
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    # eval isn't full episode if GIF_NUM_FRAMES < num_inner_steps
    raw_return_agents = jnp.zeros((env.num_agents,), dtype=jnp.float32)
    # optional: actual optimization-target team return from env reward signal
    return_team = 0.0
    
    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/cleanup"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # 获取所有智能体的观察
        # print(o_t)
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)

        # 使用模型选择动作
        network = Actor(action_dim=env.action_space().n, activation=config["ACTIVATION"])  # 使用与训练时相同的参数
        pi= network.apply(params, obs_batch)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)
        
        # 转换动作格式
        env_act = {k: v.squeeze() for k, v in unbatchify(
            actions, env.agents, 1, env.num_agents
        ).items()}
        
        # 执行动作
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]

        # ==== EVAL METRICS ====
        raw_step = info["raw_reward_individual"]  # shape (n_agents,)
        raw_return_agents = raw_return_agents + raw_step
        # changed to mean instead of sum because shared_reward (sum) already copied across all agents
        return_team += float(reward.mean())
        
        # 记录结果
        # episode_reward += sum(reward.values())
        
        # 渲染
        img = env.render(state)
        pics.append(img)
        
        # print('###################')
        # print(f'Actions: {env_act}')
        # print(f'Reward: {reward}')
        # print(f'State: {state.agent_locs}')
        # print("###################")

    # ====== CALCULATE / LOG EVAL METRICS =======
    raw_return_team = raw_return_agents.sum()
    raw_variance = jnp.var(raw_return_agents)

    # TODO: some Jax boolean index error
    # diff = jnp.abs(raw_return_agents[:, None] - raw_return_agents[None, :])  # (A, A)
    # mask = jnp.triu(jnp.ones((env.num_agents, env.num_agents), dtype=bool), k=1)
    # raw_pair_absdiff = diff[mask].mean()

    eval_metrics = {}
    for i in range(env.num_agents):
        eval_metrics[f"eval/raw_return_agent{i}"] = float(raw_return_agents[i])
    eval_metrics["eval/raw_return_team"] = float(raw_return_team)
    eval_metrics["eval/raw_return_variance"] = float(raw_variance)
    # eval_metrics["eval/raw_pairwise_absdiff"] = float(raw_pair_absdiff)
    eval_metrics["eval/opt_tgt_return_team"] = float(return_team)

    wandb.log(eval_metrics, step=int(wandb_step))

    if log_gif:
    # 保存GIF
        print(f"Saving Episode GIF")
        new_pics = [Image.fromarray(img) for img in pics]
        gif_path = f"{root_dir}/{env.num_agents}-agents_seed-{config['SEED']}_frames-{len(new_pics)}.gif"
        new_pics[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            optimize=False,
            append_images=new_pics[1:],
            duration=200,
            loop=0,
        )

        print("Logging GIF to WandB")
        wandb.log({"eval/episode_gif": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")}, step=int(wandb_step))
        
        # print(f"Episode {episode} total reward: {episode_reward}")
def tune(default_config):
    """
    Hyperparameter sweep with wandb, including logic to:
    - Initialize wandb
    - Train for each hyperparameter set
    - Save checkpoint
    - Evaluate and log GIF
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "cleanup",
        "method": "grid",
        "metric": {
            "name": "returned_episode_original_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            "ACTIVATION": {"values": ["relu", "tanh"]},
            "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            "NUM_STEPS": {"values": [64, 128, 256]},
        },
    }

    def wrapped_make_train():


        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        # only overwrite the single nested key we're sweeping
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v


        # Rename the run for clarity
        run_name = f"sweep_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0])

        # Evaluate and log
        # params = load_params(train_state.params)
        # test_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        # evaluate(params, test_env, config)

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="config", config_name="mappo_cnn_cleanup_test_mac")
def main(config):
    # if config["TUNE"]:
    #     tune(config)
    # else:
    single_run(config)

if __name__ == "__main__":
    main()
