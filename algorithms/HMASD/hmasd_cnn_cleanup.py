"""
HMASD (Hierarchical Multi-Agent Skill Discovery) for SocialJax cleanup environment.

Ported from NeurIPS2023_HMASD_code (PyTorch) to JAX/Flax.
Training loop structure mirrors MAPPO cleanup baseline.

Milestone 3: Training loop skeleton + basic env integration.
  - Nested scan: outer _skill_interval (SKILL_STEPS) x inner _env_step (SKILL_INTERVAL)
  - Correct transition shapes, GRU resets, intrinsic rewards
  - PPO updates are TODO stubs (Milestone 4)
  - evaluate() uses fixed random skills (actor-only, basic version)
"""

import os
import pickle
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from PIL import Image

import hydra
import socialjax
from socialjax.wrappers.baselines import LogWrapper

from hmasd_networks import (
    SkillActor,
    SkillCritic,
    SkillCoordinator,
    TeamDiscriminator,
    IndividualDiscriminator,
    compute_team_intrinsic_reward,
    compute_indi_intrinsic_reward,
)


# ============================================================================
# Transition NamedTuples
# ============================================================================

class LowTransition(NamedTuple):
    """Per-step transition for the low-level actor/critic (NUM_STEPS entries after reshape)."""
    global_done: jnp.ndarray       # (NUM_ACTORS,) — ep done tiled to actors, for GAE masking
    done: jnp.ndarray              # (NUM_ACTORS,) — current-step done (correct for GAE)
    action: jnp.ndarray            # (NUM_ACTORS,)
    value: jnp.ndarray             # (NUM_ACTORS,) — critic value tiled from (NUM_ENVS,)
    reward: jnp.ndarray            # (NUM_ACTORS,) — combined intrinsic+extrinsic low-level reward
    log_prob: jnp.ndarray          # (NUM_ACTORS,)
    obs: jnp.ndarray               # (NUM_ACTORS, H, W, C) — obs at this step
    rnn_state_actor: jnp.ndarray   # (NUM_ACTORS, HIDDEN_SIZE) — pre-step actor GRU
    rnn_state_critic: jnp.ndarray  # (NUM_ENVS, HIDDEN_SIZE) — pre-step critic GRU (per-env)
    team_skill_onehot: jnp.ndarray # (NUM_ACTORS, N_Z_TEAM)
    indi_skill_onehot: jnp.ndarray # (NUM_ACTORS, N_Z_INDI)
    env_reward: jnp.ndarray        # (NUM_ENVS, n_agents) — raw env reward, env-major
    info: dict                     # transposed info, values: (NUM_ACTORS,)


class HighTransition(NamedTuple):
    """Per-skill-interval transition for the high-level coordinator (SKILL_STEPS entries)."""
    done: jnp.ndarray           # (NUM_ENVS,) — ep done at end of interval
    world_state: jnp.ndarray    # (NUM_ENVS, H, W, C_ws) — world state at interval start
    all_obs: jnp.ndarray        # (NUM_ENVS, n_agents, H, W, C) — all obs at interval start
    skill_actions: jnp.ndarray  # (NUM_ENVS, n_agents+1) int32 — [Z, z1..z_n]
    value: jnp.ndarray          # (NUM_ENVS, n_agents+1) — coordinator value estimates
    log_prob: jnp.ndarray       # (NUM_ENVS, n_agents+1) — coordinator log probs
    reward: jnp.ndarray         # (NUM_ENVS, n_agents+1) — summed env reward over interval


# ============================================================================
# Batchify / Unbatchify helpers (copied from MAPPO cleanup)
# ============================================================================

def batchify(x: dict, agent_list, num_actors):
    """Stack dict values by str key, reshape to (num_actors, -1).
    Used for done dict: {str(agent): (NUM_ENVS,)} → (NUM_ACTORS, 1) → squeeze.
    """
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_numpy(x, agent_list, num_actors):
    """Stack array slices by integer index, reshape to (num_actors, -1).
    Used for reward: (NUM_ENVS, n_agents) array → (NUM_ACTORS, 1) → squeeze.
    """
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Reshape (NUM_ACTORS,) → (num_actors, num_envs, -1), return as dict.
    Note: num_actors here = n_agents (number of agents), not total actors.
    """
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


# ============================================================================
# Main training function
# ============================================================================

def make_train(config):
    # --- Env setup ---
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    n_agents = env.num_agents

    config["NUM_ACTORS"] = n_agents * config["NUM_ENVS"]
    config["SKILL_STEPS"] = config["NUM_STEPS"] // config["SKILL_INTERVAL"]
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    env = LogWrapper(env)

    # Observation and world-state shapes
    obs_shape = env.observation_space()[0].shape          # (H, W, C)
    ws_shape = (*obs_shape[:-1], obs_shape[-1] * n_agents)  # (H, W, C*n_agents)

    NUM_ENVS = config["NUM_ENVS"]
    NUM_ACTORS = config["NUM_ACTORS"]
    SKILL_INTERVAL = config["SKILL_INTERVAL"]
    SKILL_STEPS = config["SKILL_STEPS"]
    HIDDEN_SIZE = config["HIDDEN_SIZE"]
    N_Z_TEAM = config["N_Z_TEAM"]
    N_Z_INDI = config["N_Z_INDI"]

    # --- Network instantiation ---
    actor = SkillActor(
        action_dim=env.action_space().n,
        hidden_size=HIDDEN_SIZE,
        n_z_team=N_Z_TEAM,
        n_z_indi=N_Z_INDI,
        activation=config["ACTIVATION"],
    )
    critic = SkillCritic(
        hidden_size=HIDDEN_SIZE,
        n_z_team=N_Z_TEAM,
        activation=config["ACTIVATION"],
    )
    coord = SkillCoordinator(
        n_agents=n_agents,
        n_z_team=N_Z_TEAM,
        n_z_indi=N_Z_INDI,
        n_block=config["N_BLOCK"],
        n_embd=config["N_EMBD"],
        n_head=config["N_HEAD"],
        activation=config["ACTIVATION"],
    )
    team_disc = TeamDiscriminator(
        n_z_team=N_Z_TEAM,
        activation=config["ACTIVATION"],
    )
    indi_disc = IndividualDiscriminator(
        n_z_indi=N_Z_INDI,
        n_z_team=N_Z_TEAM,
        activation=config["ACTIVATION"],
    )

    # --- Dummy inputs for network init ---
    _obs_dummy = jnp.zeros((1, *obs_shape))
    _ws_dummy = jnp.zeros((1, *ws_shape))
    _team_dummy = jnp.zeros((1, N_Z_TEAM))
    _indi_dummy = jnp.zeros((1, N_Z_INDI))
    _rnn_dummy = jnp.zeros((1, HIDDEN_SIZE))
    _all_obs_dummy = jnp.zeros((1, n_agents, *obs_shape))
    _actions_dummy = jnp.zeros((1, n_agents + 1), dtype=jnp.int32)

    def _make_train_state(rng):
        """Initialize all 5 networks and create TrainState objects."""
        rng, r1, r2, r3, r4, r5 = jax.random.split(rng, 6)

        actor_params = actor.init(r1, _obs_dummy, _team_dummy, _indi_dummy, _rnn_dummy)
        critic_params = critic.init(r2, _ws_dummy, _team_dummy, _rnn_dummy)
        coord_params = coord.init(r3, _ws_dummy, _all_obs_dummy, _actions_dummy)
        team_disc_params = team_disc.init(r4, _ws_dummy)
        indi_disc_params = indi_disc.init(r5, _obs_dummy, _team_dummy)

        def _make_tx(lr, max_grad_norm):
            return optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr, eps=1e-5),
            )

        actor_ts = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=_make_tx(config["L_LR"], config["L_MAX_GRAD_NORM"]),
        )
        critic_ts = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=_make_tx(config["L_LR"], config["L_MAX_GRAD_NORM"]),
        )
        coord_ts = TrainState.create(
            apply_fn=coord.apply,
            params=coord_params,
            tx=_make_tx(config["H_LR"], config["H_MAX_GRAD_NORM"]),
        )
        team_disc_ts = TrainState.create(
            apply_fn=team_disc.apply,
            params=team_disc_params,
            tx=_make_tx(config["D_TEAM_LR"], config["D_MAX_GRAD_NORM"]),
        )
        indi_disc_ts = TrainState.create(
            apply_fn=indi_disc.apply,
            params=indi_disc_params,
            tx=_make_tx(config["D_INDI_LR"], config["D_MAX_GRAD_NORM"]),
        )

        train_states = (actor_ts, critic_ts, coord_ts, team_disc_ts, indi_disc_ts)
        return train_states, rng

    def init_runner_state(rng):
        """Initialize env, network params, and all runner state."""
        train_states, rng = _make_train_state(rng)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        # obsv: (NUM_ENVS, n_agents, H, W, C)

        rnn_actor = jnp.zeros((NUM_ACTORS, HIDDEN_SIZE))
        rnn_critic = jnp.zeros((NUM_ENVS, HIDDEN_SIZE))
        last_done = jnp.zeros((NUM_ENVS,), dtype=jnp.bool_)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, env_state, obsv, last_done, rnn_actor, rnn_critic, _rng)
        return runner_state

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    def _update_step(update_runner_state, unused):
        runner_state, update_steps = update_runner_state
        train_states, env_state, last_obs, last_done, rnn_actor, rnn_critic, rng = runner_state
        actor_ts, critic_ts, coord_ts, team_disc_ts, indi_disc_ts = train_states

        # -----------------------------------------------------------------------
        # Inner scan: 1 env step (runs SKILL_INTERVAL times per skill interval)
        # -----------------------------------------------------------------------
        def _env_step(inner_carry, unused):
            (env_state, last_obs, rnn_actor, rnn_critic,
             team_skill_idx, indi_skill_idx_actors,
             team_skill_onehot_envs, team_skill_onehot_actors, indi_skill_onehot_actors,
             rng) = inner_carry

            # 1. Batchify obs to agent-major: (NUM_ENVS, n_agents, H, W, C) → (NUM_ACTORS, H, W, C)
            obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(NUM_ACTORS, *obs_shape)

            # 2. Actor forward pass
            rng, _rng_actor = jax.random.split(rng)
            pi, new_rnn_actor = actor.apply(
                actor_ts.params,
                obs_batch, team_skill_onehot_actors, indi_skill_onehot_actors, rnn_actor
            )
            action = pi.sample(seed=_rng_actor)  # (NUM_ACTORS,)
            log_prob = pi.log_prob(action)        # (NUM_ACTORS,)

            # 3. World state per-env: (NUM_ENVS, n_agents, H, W, C) → (NUM_ENVS, H, W, C*n_agents)
            world_state = jnp.transpose(last_obs, (0, 2, 3, 1, 4)).reshape(NUM_ENVS, *ws_shape)

            # 4. Critic forward pass (per-env), tile value to per-actor (agent-major)
            value_env, new_rnn_critic = critic.apply(
                critic_ts.params,
                world_state, team_skill_onehot_envs, rnn_critic
            )  # value_env: (NUM_ENVS,)
            value_actors = jnp.tile(value_env[None, :], (n_agents, 1)).reshape(NUM_ACTORS)

            # 5. Step env (vmapped over NUM_ENVS)
            env_act = unbatchify(action, env.agents, NUM_ENVS, n_agents)
            env_act = [v for v in env_act.values()]

            rng, _rng_step = jax.random.split(rng)
            rng_step = jax.random.split(_rng_step, NUM_ENVS)
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state, env_act
            )
            # reward: (NUM_ENVS, n_agents)  [raw array from env]
            # done:   {'0': (NUM_ENVS,), ..., '__all__': (NUM_ENVS,)}
            # obsv:   (NUM_ENVS, n_agents, H, W, C)
            # info:   {key: (NUM_ENVS, n_agents)} → transpose to agent-major below

            # Transpose info to agent-major: (NUM_ENVS, n_agents) → (n_agents, NUM_ENVS) → (NUM_ACTORS,)
            info = jax.tree_util.tree_map(
                lambda x: jnp.transpose(x, (1, 0)).reshape(NUM_ACTORS),
                info,
            )

            # 6. Intrinsic rewards on NEXT state (paper Eq. 4)
            next_world_state = jnp.transpose(obsv, (0, 2, 3, 1, 4)).reshape(NUM_ENVS, *ws_shape)
            next_obs_batch = jnp.transpose(obsv, (1, 0, 2, 3, 4)).reshape(NUM_ACTORS, *obs_shape)

            # Team discriminator: (NUM_ENVS, N_Z_TEAM) logits → (NUM_ENVS,) reward
            team_logits = team_disc.apply(team_disc_ts.params, next_world_state)
            team_intri_env = compute_team_intrinsic_reward(team_logits, team_skill_idx)
            team_intri_actors = jnp.tile(team_intri_env[None, :], (n_agents, 1)).reshape(NUM_ACTORS)

            # Individual discriminator: (NUM_ACTORS, N_Z_INDI) logits → (NUM_ACTORS,) reward
            indi_logits = indi_disc.apply(indi_disc_ts.params, next_obs_batch, team_skill_onehot_actors)
            indi_intri = compute_indi_intrinsic_reward(indi_logits, indi_skill_idx_actors)

            # 7. Combined low-level reward
            env_rew_actors = batchify_numpy(reward, env.agents, NUM_ACTORS).squeeze()
            combined_reward = (
                config["LAMBDA_ENV"] * env_rew_actors
                + config["LAMBDA_TEAM"] * team_intri_actors
                + config["LAMBDA_INDI"] * indi_intri
            )

            # 8. Extract done signals; GRU reset on episode done
            ep_done = done["__all__"]                               # (NUM_ENVS,)
            done_batch = batchify(done, env.agents, NUM_ACTORS).squeeze()  # (NUM_ACTORS,)

            # Reset GRU hidden states where episode finished
            new_rnn_actor = new_rnn_actor * (1.0 - done_batch[:, None])
            new_rnn_critic = new_rnn_critic * (1.0 - ep_done[:, None])

            # 9. Build LowTransition (pre-step rnn_states stored for PPO Milestone 4)
            transition = LowTransition(
                global_done=jnp.tile(ep_done, n_agents),  # (NUM_ACTORS,) agent-major
                done=done_batch,
                action=action,
                value=value_actors,
                reward=combined_reward,
                log_prob=log_prob,
                obs=obs_batch,
                rnn_state_actor=rnn_actor,   # pre-step
                rnn_state_critic=rnn_critic,  # pre-step (per-env)
                team_skill_onehot=team_skill_onehot_actors,
                indi_skill_onehot=indi_skill_onehot_actors,
                env_reward=reward,           # (NUM_ENVS, n_agents)
                info=info,
            )

            inner_carry = (
                env_state, obsv, new_rnn_actor, new_rnn_critic,
                team_skill_idx, indi_skill_idx_actors,
                team_skill_onehot_envs, team_skill_onehot_actors, indi_skill_onehot_actors,
                rng,
            )
            return inner_carry, transition

        # -----------------------------------------------------------------------
        # Outer scan body: 1 skill interval = SKILL_INTERVAL env steps
        # -----------------------------------------------------------------------
        def _skill_interval(outer_carry, unused):
            env_state, last_obs, last_done, rnn_actor, rnn_critic, rng = outer_carry

            # World state and all_obs at start of interval (for coordinator)
            world_state_coord = jnp.transpose(last_obs, (0, 2, 3, 1, 4)).reshape(NUM_ENVS, *ws_shape)
            all_obs_coord = last_obs  # (NUM_ENVS, n_agents, H, W, C)

            # 1. Coordinator: autoregressive skill assignment
            rng, _rng_coord = jax.random.split(rng)
            skill_actions, skill_log_probs, skill_values = coord.apply(
                coord_ts.params,
                world_state_coord, all_obs_coord, _rng_coord,
                method=coord.get_actions,
            )
            # skill_actions: (NUM_ENVS, n_agents+1) int32 — [Z, z1..z_n]
            team_skill_idx = skill_actions[:, 0]    # (NUM_ENVS,)
            indi_skill_idx = skill_actions[:, 1:]   # (NUM_ENVS, n_agents)

            # 2. Prepare skill tensors for inner scan carry
            team_skill_onehot_envs = jax.nn.one_hot(team_skill_idx, N_Z_TEAM)  # (NUM_ENVS, N_Z_TEAM)

            # Tile team skill to agent-major: each agent in each env gets the team skill
            team_skill_onehot_actors = jnp.tile(
                team_skill_onehot_envs[None, :, :], (n_agents, 1, 1)
            ).reshape(NUM_ACTORS, N_Z_TEAM)

            # Individual skills: (NUM_ENVS, n_agents) → agent-major (NUM_ACTORS,)
            indi_skill_idx_actors = jnp.transpose(indi_skill_idx, (1, 0)).reshape(NUM_ACTORS)

            # Individual skill one-hots: (NUM_ENVS, n_agents, N_Z_INDI) → agent-major
            indi_skill_onehot = jax.nn.one_hot(indi_skill_idx, N_Z_INDI)  # (NUM_ENVS, n_agents, N_Z_INDI)
            indi_skill_onehot_actors = jnp.transpose(indi_skill_onehot, (1, 0, 2)).reshape(NUM_ACTORS, N_Z_INDI)

            # 3. Inner scan: collect SKILL_INTERVAL steps
            inner_carry_init = (
                env_state, last_obs, rnn_actor, rnn_critic,
                team_skill_idx, indi_skill_idx_actors,
                team_skill_onehot_envs, team_skill_onehot_actors, indi_skill_onehot_actors,
                rng,
            )
            inner_carry, l_traj = jax.lax.scan(
                _env_step, inner_carry_init, None, SKILL_INTERVAL
            )
            # l_traj: LowTransition with leading dim SKILL_INTERVAL

            # Unpack inner carry (env_state, obs updated; rnn states updated in place)
            env_state, last_obs, rnn_actor, rnn_critic, _, _, _, _, _, rng = inner_carry

            # 4. High-level done: episode done at end of interval
            # global_done is tile(ep_done, n_agents); first NUM_ENVS = agent 0's ep_done
            last_ep_done = l_traj.global_done[-1, :NUM_ENVS]  # (NUM_ENVS,)

            # 5. High-level reward: sum env reward over interval, mean across agents
            # l_traj.env_reward: (SKILL_INTERVAL, NUM_ENVS, n_agents)
            h_reward_env = l_traj.env_reward.sum(axis=0).mean(axis=-1)  # (NUM_ENVS,)
            h_reward = jnp.tile(h_reward_env[:, None], (1, n_agents + 1))  # (NUM_ENVS, n_agents+1)

            # 6. Build HighTransition
            h_transition = HighTransition(
                done=last_ep_done,
                world_state=world_state_coord,
                all_obs=all_obs_coord,
                skill_actions=skill_actions,
                value=skill_values,
                log_prob=skill_log_probs,
                reward=h_reward,
            )

            outer_carry = (env_state, last_obs, last_ep_done, rnn_actor, rnn_critic, rng)
            return outer_carry, (h_transition, l_traj)

        # -----------------------------------------------------------------------
        # Run outer scan: SKILL_STEPS skill intervals
        # -----------------------------------------------------------------------
        outer_carry_init = (env_state, last_obs, last_done, rnn_actor, rnn_critic, rng)
        outer_carry, (h_traj, l_traj) = jax.lax.scan(
            _skill_interval, outer_carry_init, None, SKILL_STEPS
        )
        env_state, last_obs, last_done, rnn_actor, rnn_critic, rng = outer_carry

        # Flatten l_traj: (SKILL_STEPS, SKILL_INTERVAL, ...) → (NUM_STEPS, ...)
        l_traj = jax.tree_util.tree_map(
            lambda x: x.reshape((config["NUM_STEPS"],) + x.shape[2:]),
            l_traj,
        )
        # h_traj shape stays (SKILL_STEPS, NUM_ENVS, ...)

        # -----------------------------------------------------------------------
        # PPO updates (Milestone 4 TODO stubs)
        # -----------------------------------------------------------------------
        # TODO (Milestone 4): Low-level GAE + PPO update for actor/critic
        # TODO (Milestone 4): High-level GAE + PPO update for coordinator
        # TODO (Milestone 4): Discriminator cross-entropy updates for team_disc/indi_disc
        # train_states pass through unchanged for now

        # -----------------------------------------------------------------------
        # Metrics and WandB logging
        # -----------------------------------------------------------------------
        def _reduce_metric_dict(m):
            """Reduce per-step per-actor metric dict to loggable scalars.

            Matches MAPPO cleanup _reduce_metric_dict pattern.
            m values have shape (NUM_STEPS, NUM_ACTORS) after the nested scan reshape.
            """
            out = {}
            episode_keys = ("returned_episode_returns", "returned_episode_lengths", "returned_episode")
            episode_mask = (m["returned_episode"] > 0).astype(jnp.float32)
            denom = episode_mask.sum()

            def episodic_mean(val):
                num = (val * episode_mask).sum()
                return jnp.where(denom > 0, num / denom, jnp.nan)

            if "raw_reward_individual" in m:
                raw = m["raw_reward_individual"]  # (NUM_STEPS, NUM_ACTORS)
                if raw.ndim == 2:
                    raw = raw.reshape((raw.shape[0], n_agents, NUM_ENVS))
                ep_returns_agent_env = raw.sum(axis=0)              # (n_agents, NUM_ENVS)
                mean_ep_return_per_agent = ep_returns_agent_env.mean(axis=1)  # (n_agents,)
                mean_ep_return_team = ep_returns_agent_env.sum(axis=0).mean()  # scalar
                mean_ep_return_variance = ep_returns_agent_env.var(axis=0).mean()  # scalar

                for i in range(n_agents):
                    out[f"rollout/raw_ep_return_agent{i}"] = mean_ep_return_per_agent[i]
                out["rollout/raw_ep_return_team"] = mean_ep_return_team
                out["rollout/raw_ep_return_variance"] = mean_ep_return_variance

            for k, v in m.items():
                if k == "raw_reward_individual":
                    continue  # already handled above
                if k in episode_keys:
                    out[k] = episodic_mean(v)
                elif k == "clean_action_info":
                    out["rollout/clean_action_chunk"] = v.sum()
                elif k == "cleaned_water":
                    out["rollout/cleaned_water_mean"] = v.mean()
                    out["rollout/cleaned_water_final"] = v[-1].mean()
                else:
                    out[k] = v.mean()

            return out

        update_steps = update_steps + 1
        metric = _reduce_metric_dict(l_traj.info)

        # HMASD-specific rollout metrics
        metric["rollout/combined_reward_mean"] = l_traj.reward.mean()
        metric["rollout/env_reward_mean"] = l_traj.env_reward.mean()
        # Note: individual intrinsic reward breakdown (team vs indi) not stored in LowTransition.
        # Full breakdown can be added in Milestone 5 when needed for analysis.

        metric["update_step"] = update_steps
        metric["env_step"] = update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]

        def callback(metric):
            episodic_keys = ("returned_episode_returns", "returned_episode_lengths", "returned_episode")
            filtered_metric = {}
            for k, v in metric.items():
                if k in episodic_keys and np.isnan(np.asarray(v)).any():
                    continue
                filtered_metric[k] = v
            wandb.log(filtered_metric, step=metric["env_step"])

        jax.debug.callback(callback, metric)

        runner_state = (train_states, env_state, last_obs, last_done, rnn_actor, rnn_critic, rng)
        return (runner_state, update_steps), metric

    # -----------------------------------------------------------------------
    # Chunk pattern (10 eval chunks + remainder, matching MAPPO)
    # -----------------------------------------------------------------------
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


# ============================================================================
# Evaluate (basic actor-only with fixed random skills, Milestone 3 version)
# ============================================================================

def evaluate(actor_params, env, save_path, config, wandb_step: int, log_gif: bool = False):
    """Basic eval: single env, fixed random skills for full episode, actor with GRU.

    Uses random fixed skills (not coordinator) — coordinator eval deferred to Milestone 5.
    """
    HIDDEN_SIZE = config["HIDDEN_SIZE"]
    N_Z_TEAM = config["N_Z_TEAM"]
    N_Z_INDI = config["N_Z_INDI"]
    n_agents = env.num_agents

    actor_net = SkillActor(
        action_dim=env.action_space().n,
        hidden_size=HIDDEN_SIZE,
        n_z_team=N_Z_TEAM,
        n_z_indi=N_Z_INDI,
        activation=config["ACTIVATION"],
    )

    rng = jax.random.PRNGKey(0)

    # Sample fixed random skills for the whole episode
    rng, _rng_team, _rng_indi = jax.random.split(rng, 3)
    team_skill_idx = jax.random.randint(_rng_team, shape=(), minval=0, maxval=N_Z_TEAM)
    indi_skill_idx = jax.random.randint(_rng_indi, shape=(n_agents,), minval=0, maxval=N_Z_INDI)

    team_skill_onehot = jax.nn.one_hot(team_skill_idx, N_Z_TEAM)           # (N_Z_TEAM,)
    team_skill_batch = jnp.tile(team_skill_onehot[None, :], (n_agents, 1)) # (n_agents, N_Z_TEAM)
    indi_skill_batch = jax.nn.one_hot(indi_skill_idx, N_Z_INDI)            # (n_agents, N_Z_INDI)

    # Initial GRU state: (n_agents, HIDDEN_SIZE) for single-env eval
    rnn_state = jnp.zeros((n_agents, HIDDEN_SIZE))

    rng, _rng_reset = jax.random.split(rng)
    obs, state = env.reset(_rng_reset)
    done = False

    raw_return_agents = jnp.zeros((n_agents,), dtype=jnp.float32)
    return_team = 0.0

    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = "evaluation/cleanup"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for _ in range(config["GIF_NUM_FRAMES"]):
        # obs: (n_agents, H, W, C) — array indexed by integer agent id
        obs_batch = jnp.stack([obs[a] for a in env.agents])  # (n_agents, H, W, C)

        pi, rnn_state = actor_net.apply(
            actor_params, obs_batch, team_skill_batch, indi_skill_batch, rnn_state
        )
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)  # (n_agents,)

        env_act = {k: v.squeeze() for k, v in unbatchify(
            actions, env.agents, 1, n_agents
        ).items()}

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(
            _rng, state, [v.item() for v in env_act.values()]
        )
        done = done["__all__"]

        # Accumulate raw individual rewards
        raw_step = info["raw_reward_individual"]  # (n_agents,) from cleanup env
        raw_return_agents = raw_return_agents + raw_step
        return_team += float(reward.mean())

        # Reset GRU if episode done
        rnn_state = rnn_state * (1.0 - float(done))

        img = env.render(state)
        pics.append(img)

    # Log eval metrics
    raw_return_team = raw_return_agents.sum()
    raw_variance = jnp.var(raw_return_agents)

    eval_metrics = {}
    for i in range(n_agents):
        eval_metrics[f"eval/raw_return_agent{i}"] = float(raw_return_agents[i])
    eval_metrics["eval/raw_return_team"] = float(raw_return_team)
    eval_metrics["eval/raw_return_variance"] = float(raw_variance)
    eval_metrics["eval/opt_tgt_return_team"] = float(return_team)

    wandb.log(eval_metrics, step=int(wandb_step))

    if log_gif:
        print("Saving Episode GIF")
        new_pics = [Image.fromarray(img) for img in pics]
        gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_frames-{len(new_pics)}.gif"
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
        wandb.log(
            {"eval/episode_gif": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")},
            step=int(wandb_step),
        )


# ============================================================================
# Single run (WandB init + chunk loop + eval)
# ============================================================================

def single_run(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["HMASD", "cleanup"],
        group=config["WANDB_GROUP"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"hmasd_cnn_cleanup",
    )

    init_runner_state, train_chunk, remainder_chunk, chunk_updates, remainder_updates = make_train(config)
    init_jit = jax.jit(init_runner_state)
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
        update_runner_state, _ = chunk_jit(update_runner_state)

        # Extract actor params for eval
        train_states = update_runner_state[0][0]
        actor_params = train_states[0].params

        update_step = int(update_runner_state[1])
        env_step = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

        evaluate(
            actor_params,
            socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
            save_path=None,
            config=config,
            wandb_step=env_step,
            log_gif=False,
        )

    # Remainder chunk
    if remainder_updates > 0:
        update_runner_state, _ = remainder_jit(update_runner_state)

    # Final eval with GIF
    train_states = update_runner_state[0][0]
    actor_params = train_states[0].params
    update_step = int(update_runner_state[1])
    env_step = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

    evaluate(
        actor_params,
        socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
        save_path=None,
        config=config,
        wandb_step=env_step,
        log_gif=True,
    )
    print("Finished training and evals")


# ============================================================================
# Hydra entry point
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="hmasd_cnn_cleanup")
def main(config):
    single_run(config)


if __name__ == "__main__":
    main()
