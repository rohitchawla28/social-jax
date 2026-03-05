""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
"""

import sys
# sys.path.append('/home/shuqing/SocialJax')
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
# from flax.training import checkpoints
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper, SVOLogWrapper
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
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(params, config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["PARAMETER_SHARING"]:
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    else:
        network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    for o in range(config["GIF_NUM_FRAMES"]):
        print(o)
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
        if config["PARAMETER_SHARING"]: 
            pi, value = network.apply(params, obs_batch)
            action = pi.sample(seed=key_a0)
            env_act = unbatchify(
                action, env.agents, 1, env.num_agents
            )           
        else:
            env_act = {}
            for i in range(env.num_agents):
                pi, value = network[i].apply(params[i], obs_batch)
                action = pi.sample(seed=key_a0)
                env_act[env.agents[i]] = action


        

        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors):
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
    if config["PARAMETER_SHARING"]:
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=0.,
        end_value=1.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    rew_shaping_anneal_org = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # INIT NETWORK
    if config["PARAMETER_SHARING"]:
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    else:
        network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
    
    init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
    
    # helper method for init_runner_state() to create TrainState object
    def _make_train_state(rng):
        rng, _rng = jax.random.split(rng)
        if config["PARAMETER_SHARING"]:
            network_params = network.init(_rng, init_x)
        else:
            network_params = [network[i].init(_rng, init_x) for i in range(env.num_agents)]

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        if config["PARAMETER_SHARING"]:
            train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        else:
            train_state = [
                TrainState.create(apply_fn=network[i].apply, params=network_params[i], tx=tx)
                for i in range(env.num_agents)
            ]
        return train_state, rng

    # new method for chunk refactor
    # runner state init method called once only
    def init_runner_state(rng):
        train_state, rng = _make_train_state(rng)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        rng, _rng = jax.random.split(rng)

        return (train_state, env_state, obsv, 0, _rng)

    # TRAIN LOOP
    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
        # _env_step runs 1 step and collects transition data 
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, update_step, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)
            if config["PARAMETER_SHARING"]:
                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                # print("input_obs_shape", obs_batch.shape)
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
            else:
                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                env_act = {}
                log_prob = []
                value = []
                for i in range(env.num_agents):
                    # print("input_obs_shape", obs_batch[i].shape)
                    pi, value_i = network[i].apply(train_state[i].params, obs_batch[i])
                    action = pi.sample(seed=_rng)
                    log_prob.append(pi.log_prob(action))
                    env_act[env.agents[i]] = action
                    value.append(value_i)

            # env_act = {k: v.flatten() for k, v in env_act.items()}
            env_act = [v for v in env_act.values()]
            
            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])

            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)

            # info object AT THIS POINT is env-major
            # info shapes x16 for T: {  'eat_own_coins': (Array(4, dtype=int32), Array(2, dtype=int32)), 
                            # 'raw_reward_individual': (Array(4, dtype=int32), Array(2, dtype=int32)), 
                            # 'returned_episode': (Array(4, dtype=int32), Array(2, dtype=int32)), 
                            # 'returned_episode_lengths': (Array(4, dtype=int32), Array(2, dtype=int32)), 
                            # 'returned_episode_returns': (Array(4, dtype=int32), Array(2, dtype=int32))}


            # current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
            # shaped_reward = compute_grouped_rewards(reward)
            # reward = jax.tree_util.tree_map(lambda x,y: x*rew_shaping_anneal_org(current_timestep)+y*rew_shaping_anneal(current_timestep), reward, shaped_reward)

            
            if config["PARAMETER_SHARING"]:
                # I'm keeping previous env-major flattening for reference - (NUM_ENVS, num_agents) -> (NUM_ACTORS,)
                # This can misalign with obs,reward,done actor indexing when PARAMETER_SHARING=True
                    # info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                # Correct actor alignment is below:
                    # Obs_batch is built with transpose(last_obs, (1, 0, ...)).reshape(-1, ...) 
                    # so actor axis is agent-major: actor = agent_id * NUM_ENVS + env_id.
                    
                    # Vmapped env.step returns per-key info as (NUM_ENVS, num_agents) which is env-major.
                    # Transpose to (num_agents, NUM_ENVS) before flattening so info matches obs/reward/done.
                    # flattened info is then env0 agent0, env1 agent0, ..., envN agent0, env0 agent1, ..., envN agent1
                info = jax.tree_util.tree_map(
                    lambda x: jnp.transpose(x, (1, 0)).reshape((config["NUM_ACTORS"],)),
                    info,
                )
                transition = Transition(
                    batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                    )
            else:
                transition = []
                # changed to done_list from done to not overload the term
                done_list = [v for v in done.values()]
                for i in range(env.num_agents):
                    info_i = {key: jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"]),1), value[:,i]) for key, value in info.items()}
                    transition.append(Transition(
                        done_list[i],
                        env_act[i],
                        value[i],
                        reward[:,i],
                        log_prob[i],
                        obs_batch[i],
                        info_i,
                    ))
            runner_state = (train_state, env_state, obsv, update_step, rng)
            return runner_state, transition

        # collect trajectories for 1 rollout batch/update_step (which has NUM_STEPS=1000 in it)
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        # ===== some debugging prints =====

        # do_print = (runner_state[3] == 0)
        # jax.lax.cond(
        #     do_print,
        #     lambda _: jax.debug.print(
        #         "traj_batch: {t}, class: {cl}, type: {ty}\n\n",
        #         t=traj_batch,
        #         ty=type(traj_batch)
        #     ),
        #     lambda _: None,
        #     operand=None,
        # )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, update_step, rng = runner_state
        if config["PARAMETER_SHARING"]:
            last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
            _, last_val = network.apply(train_state.params, last_obs_batch)
        else:
            last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
            last_val = []
            for i in range(env.num_agents):
                _, last_val_i = network[i].apply(train_state[i].params, last_obs_batch[i])
                last_val.append(last_val_i)
            last_val = jnp.stack(last_val, axis=0)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = transition.done, transition.value, transition.reward
                # reward_mean = jnp.mean(reward, axis=0)
                # # reward_std = jnp.std(reward, axis=0) + 1e-8
                # reward = (reward - reward_mean)# / reward_std
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae
            
            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value
        
        if config["PARAMETER_SHARING"]:
            advantages, targets = _calculate_gae(traj_batch, last_val)
        else:
            advantages = []
            targets = []
            for i in range(env.num_agents):
                advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i])
                advantages.append(advantages_i)
                targets.append(targets_i)
            advantages = jnp.stack(advantages, axis=0)
            targets = jnp.stack(targets, axis=0)
        # UPDATE NETWORK
        def _update_epoch(update_state, unused, i):
            def _update_minbatch(train_state, batch_info, network_used):
                # minibatches is passed in as batch_info param?
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets, network_used):
                    # RERUN NETWORK
                    pi, value = network_used.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)
                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()

                    total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy

                    # log critic prediction value as well for analyzing training dynamics
                    value_mean = value.mean()

                    # returned as pair for jax.value_and_grad() function,
                    # where 1st elem (total_loss) is main scalar for gradient calculation
                    # and 2nd elem is auxiliary data
                    return total_loss, (value_loss, loss_actor, entropy, value_mean)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                # changed to actually extract the auxiliary metrics along w/ total_loss
                (total_loss, (value_loss, loss_actor, entropy, value_mean)), grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets, network_used
                )

                # calculate gradient norm for additional training metric
                grad_norm = optax.global_norm(grads)

                train_state = train_state.apply_gradients(grads=grads)
                train_metrics = {
                    "train/total_loss": total_loss,
                    "train/value_loss": value_loss,
                    "train/value_mean": value_mean,
                    "train/entropy": entropy,
                    "train/grad_norm": grad_norm,
                }
                return train_state, train_metrics

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            # MINIBATCH_SIZE == 1024
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            
            # batch_size (for 1 rollout batch) == 1000 * 512
            # assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"], "batch size must be equal to number of steps * number of actors"

            # shuffle the batch before splitting into minibatches
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
            # if config["PARAMETER_SHARING"]:
                
            # else:
            #     batch = jax.tree_util.tree_map(
            #         lambda x: x.reshape((batch_size,) + x.shape[2:]),  # 保持第一个维度为batch_size，自动计算第二个维度
            #         batch
            #     )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )

            # reshape into minibatches
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            # 1 epoch update: run minibatches through the network and update parameters
            if config["PARAMETER_SHARING"]:
                train_state, train_metrics = jax.lax.scan(
                    lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                )
            else:
                train_state, train_metrics = jax.lax.scan(
                    lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                )

            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, train_metrics
        
        if config["PARAMETER_SHARING"]:
            update_state = (train_state, traj_batch, advantages, targets, rng)

            # run multiple epochs on same rollout batch
            update_state, loss_info = jax.lax.scan(
                lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            # avg the training metrics over epochs and minibatches to get 1 scalar per metric each update_step
            # this matches their logging granularity
            train_metric = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = {**metric, **train_metric}
            rng = update_state[-1]
        else:
            update_state_dict = []
            metric = []
            for i in range(env.num_agents):
                update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                )
                update_state_dict.append(update_state)
                train_state[i] = update_state[0]
                metric_i = traj_batch[i].info
                train_metric_i = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
                metric_i = {**metric_i, **train_metric_i}
                metric.append(metric_i)
                rng = update_state[-1]
            
        def callback(metric):
            wandb.log(metric, step=metric["env_step"])

        update_step = update_step + 1

        # REMOVED bc avg across all timesteps wasnt good
            # write metrics to wandb after every update step / rollout
            # metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
        
        # NOTE: some fields (ex: returned_episode_returns) are episodic and only become non-zero at episode end
        def _reduce_metric_dict(m):
            """Reduce per-update metrics.
            - For fixed T CoinGame (1 episode per rollout): use last timestep for episodic signals
            - For raw per-step rewards: sum over time to get episode returns
            """
            episode_keys = ("returned_episode_returns", "returned_episode_lengths", "returned_episode")
            out = {}
            if "raw_reward_individual" in m:
                raw = m["raw_reward_individual"]                                        # shape=(T, NUM_ACTORS) after flatten

                # unflatten to (T, num_agents, NUM_ENVS) assuming agent-major flattening
                if raw.ndim == 2:
                    raw = raw.reshape((raw.shape[0], env.num_agents, config["NUM_ENVS"]))

                # episode return per agent per env
                ep_returns_agent_env = raw.sum(axis=0)                                  # (num_agents, NUM_ENVS)

                # avg across envs
                mean_ep_return_per_agent = ep_returns_agent_env.mean(axis=1)            # (num_agents,)

                # sum across agents to get team return, then avg across envs
                mean_ep_return_team = ep_returns_agent_env.sum(axis=0).mean()            # scalar

                # fairness (variance) by getting variance between agents, then avg across envs
                mean_ep_return_variance = ep_returns_agent_env.var(axis=0).mean()      # scalar

                # fairness (pairwise absolute difference)
                mean_ep_pairwise_absdiff = jnp.abs(ep_returns_agent_env[0] - ep_returns_agent_env[1]).mean()

                    # General case for num_agents > 2
                        # diff = jnp.abs(ep_returns_agent_env[:, None, :] - ep_returns_agent_env[None, :, :])  # (A, A, E)
                        # # take only upper triangle pairs i<j
                        # A = ep_ret_agent_env.shape[0]
                        # mask = jnp.triu(jnp.ones((A, A), dtype=bool), k=1)[:, :, None]               # (A, A, 1)
                        # pairwise_absdiff_mean = diff[mask].mean()                                    # scalar

                for i in range(env.num_agents):
                    out[f"rollout/raw_ep_return_agent{i}"] = mean_ep_return_per_agent[i]

                out["rollout/raw_ep_return_team"] = mean_ep_return_team
                out["rollout/raw_ep_return_variance"] = mean_ep_return_variance
                out["rollout/raw_ep_pairwise_absdiff"] = mean_ep_pairwise_absdiff

            for k, v in m.items():
                if k == "raw_reward_individual":
                    # don't want to log per-step raw rewards
                    continue
                
                # v has shape=(T, NUM_ACTORS)
                if k in episode_keys:
                    # since one episode per rollout, last timestep contains episode return value
                    out[k] = v[-1].mean()                    # v[-1].shape=(NUM_ACTORS,), then avg across actors for scalar
                else:
                    out[k] = v.mean()
            return out

        if config["PARAMETER_SHARING"]:
            metric = _reduce_metric_dict(metric)
        else:
            metric = [_reduce_metric_dict(m_i) for m_i in metric]

        if config["PARAMETER_SHARING"]:
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
        else:
            for i in range(env.num_agents):
                metric[i]["update_step"] = update_step
                metric[i]["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric = metric[0]
        metric["update_step"] = update_step
        metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
        metric["eat_own_coins"] = metric["eat_own_coins"] * config["ENV_KWARGS"]["num_inner_steps"]

        jax.debug.callback(callback, metric)

        runner_state = (train_state, env_state, last_obs, update_step, rng)
        return runner_state, metric
    
    num_evals = 10
    chunk_updates = max(1, config["NUM_UPDATES"] // num_evals)
    remainder_updates = config["NUM_UPDATES"] - (num_evals * chunk_updates)
    
    def train_chunk(runner_state):
        return jax.lax.scan(_update_step, runner_state, None, chunk_updates)
    
    if remainder_updates > 0:
        def remainder_chunk(runner_state):
            return jax.lax.scan(_update_step, runner_state, None, remainder_updates)
    else:
        remainder_chunk = None

    return init_runner_state, train_chunk, remainder_chunk, chunk_updates, remainder_updates

def single_run(config):
    config = OmegaConf.to_container(config)
    # layout_name = copy.deepcopy(config["ENV_KWARGS"]["layout"])
    # config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO"],
        group=config["WANDB_GROUP"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_cnn_coins'
    )

    # W/O REFACTOR
        # train_jit = jax.jit(make_train(config))
        # out = jax.vmap(train_jit)(rngs)

    # ===== CHUNKING REFACTOR ======
    init_runner_state, train_chunk, remainder_chunk, chunk_updates, remainder_updates = make_train(config)
    init_jit = jax.jit(init_runner_state)

    # chunk_jit is a scan for _update_step() over chunk_updates == 3906 // 10 = 390 updates
    chunk_jit = jax.jit(train_chunk)

    remainder_jit = None
    if remainder_chunk is not None:
        remainder_jit = jax.jit(remainder_chunk)    

    # key for the whole run, then split for each seed
    # removed the jax.random.split(rng, config["NUM_SEEDS"]) since we only run 1 seed for now
    rng = jax.random.PRNGKey(config["SEED"])
    print(f"config seed: {config['SEED']}")
    runner_state = init_jit(rng)

    num_evals = 10
    for k in range(num_evals):
        runner_state, _ = chunk_jit(runner_state)

        # extract params for eval
        train_state = runner_state[0]
        if config["PARAMETER_SHARING"]:
            params = train_state.params
        else:
            params = [ts.params for ts in train_state]

        update_step = int(runner_state[3])
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
        # print(f"Remainder updates > 0, == {remainder_updates}")
        runner_state, _ = remainder_jit(runner_state)

    # final eval after all updates, only eval that logs GIF
    train_state = runner_state[0]
    if config["PARAMETER_SHARING"]:
        params = train_state.params
    else:
        params = [ts.params for ts in train_state]
    
    update_step = int(runner_state[3])
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

    # Removed b/c doing new chunking refactor for eval every 10%, also don't need to save params
        # print("** Saving Results **")
        # filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}'
        # train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
        # save_path = f"./checkpoints/individual/{filename}.pkl"
        # if config["PARAMETER_SHARING"]:
        #     save_path = f"./checkpoints/individual/{filename}.pkl"
        #     save_params(train_state, save_path)
        #     params = load_params(save_path)
        # else:
        #     params = []
        #     for i in range(config['ENV_KWARGS']['num_agents']):
        #         save_path = f"./checkpoints/individual/{filename}_{i}.pkl"
        #         save_params(train_state[i], save_path)
        #         params.append(load_params(save_path))
        # evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)

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
    # new rng key for eval, seed=0 for reproducibility across eval runs
    rng = jax.random.PRNGKey(0)
    
    # each time this line occurs, master key is reassigned (each aspect of eval has unique randomness, but overall eval run is fully deterministic)
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    # eval isn't full episode if GIF_NUM_FRAMES < num_inner_steps
    raw_return_agents = jnp.zeros((env.num_agents,), dtype=jnp.float32)
    # optional actual training signal team return. this should be the 2x the raw_ep_return_team since they multiplied the reward by 2
    return_team = 0.0
    
    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/coins"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # 获取所有智能体的观察
        # print(o_t)
        # 使用模型选择动作
        if config["PARAMETER_SHARING"]:
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
            network = ActorCritic(action_dim=env.action_space().n, activation="relu")  # 使用与训练时相同的参数
            pi, _ = network.apply(params, obs_batch)
            rng, _rng = jax.random.split(rng)
            actions = pi.sample(seed=_rng)
            # 转换动作格式
            env_act = {k: v.squeeze() for k, v in unbatchify(
                actions, env.agents, 1, env.num_agents
            ).items()}
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            network = [ActorCritic(action_dim=env.action_space().n, activation="relu") for _ in range(env.num_agents)]
            for i in range(env.num_agents):
                obs = jnp.expand_dims(obs_batch[i],axis=0)
                pi, _ = network[i].apply(params[i], obs)
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action

        
        # 执行动作
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]

        # ==== EVAL METRICS ===
        # raw per-agent reward (logging only)
        raw_step = info["raw_reward_individual"]  # shape (n_agents,)
        raw_return_agents = raw_return_agents + raw_step

        return_team += float(sum(reward))
        
        # 记录结果
        # episode_reward += sum(reward.values())
        
        # 渲染
        img = env.render(state)
        pics.append(img)
        
        # print(f'Actions: {env_act}')
        # print(f'Reward: {reward}')
        # print(f'State: {state.agent_locs}')
        # print(f'State: {state.claimed_indicator_time_matrix}')

    # ====== CALCULATE / LOG EVAL METRICS =======
    raw_return_team = raw_return_agents.sum()

    # fairness variance across agents (single env episode)
    raw_variance = jnp.var(raw_return_agents)

    # pairwise abs diff
    if env.num_agents == 2:
        raw_pair_absdiff = jnp.abs(raw_return_agents[0] - raw_return_agents[1])
    # when num_agents > 2
        # else:
        #     diff = jnp.abs(raw_ep_return_agents[:, None] - raw_ep_return_agents[None, :])  # (A, A)
        #     mask = jnp.triu(jnp.ones((env.num_agents, env.num_agents), dtype=bool), k=1)
        #     fair_pair_absdiff = diff[mask].mean()

    eval_metrics = {}
    for i in range(env.num_agents):
        eval_metrics[f"eval/raw_return_agent{i}"] = float(raw_return_agents[i])
    eval_metrics["eval/raw_return_team"] = float(raw_return_team)
    eval_metrics["eval/raw_return_variance"] = float(raw_variance)
    eval_metrics["eval/raw_pairwise_absdiff"] = float(raw_pair_absdiff)
    eval_metrics["eval/opt_tgt_return_team"] = float(return_team)

    wandb.log(eval_metrics, step=int(wandb_step))
    
    if log_gif:
        print(f"Saving Episode GIF")
        new_pics = [Image.fromarray(np.array(img)) for img in pics]
        gif_path = f"{root_dir}/{env.num_agents}-agents_seed-{config['SEED']}_frames-{len(new_pics)}.gif"
        new_pics[0].save(gif_path, format="GIF", save_all=True, optimize=False, append_images=new_pics[1:], duration=200, loop=0)

        print("Logging GIF to WandB")
        if log_gif:
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
        "name": "coins",
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            # "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            # "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            # "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            # "NUM_STEPS": {"values": [64, 128, 256]},
            # "ENV_KWARGS.svo_w": {"values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            # "ENV_KWARGS.svo_ideal_angle_degrees": {"values": [0, 45, 90]},
            "SEED": {"values": [42, 52, 62]},

        },
    }

    def wrapped_make_train():


        wandb.init(
            project=default_config["PROJECT"],
            group=default_config["WANDB_GROUP"],
        )
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


@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_coins")
def main(config):
    # if config["TUNE"]:
    #     tune(config)
    # else:
    print(f"Starting single run ippo main with config:\n\n {config}\n\n")
    single_run(config)
if __name__ == "__main__":
    main()
