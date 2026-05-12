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
    world_state: jnp.ndarray       # (NUM_ENVS, H, W, C_ws) — pre-step world state for critic PPO re-eval
    team_skill_onehot: jnp.ndarray # (NUM_ACTORS, N_Z_TEAM)
    indi_skill_onehot: jnp.ndarray # (NUM_ACTORS, N_Z_INDI)
    env_reward: jnp.ndarray        # (NUM_ENVS, n_agents) — raw env reward, env-major
    team_intri: jnp.ndarray        # (NUM_ACTORS,) — team intrinsic reward, same value tiled across agents per env
    indi_intri: jnp.ndarray        # (NUM_ACTORS,) — individual intrinsic reward per actor
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

        if config.get("ANNEAL_LR", False):
            # Linear LR decay matching MAPPO convention: anneals over NUM_UPDATES outer iterations.
            # count is gradient steps; divide by (minibatches * epochs) to convert to update count.
            def _l_schedule(count):
                frac = 1.0 - (count // (config["L_NUM_MINIBATCHES"] * config["L_UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
                return config["L_LR"] * frac
            def _h_schedule(count):
                frac = 1.0 - (count // (config["H_NUM_MINIBATCHES"] * config["H_UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
                return config["H_LR"] * frac
            l_lr = _l_schedule
            h_lr = _h_schedule
        else:
            l_lr = config["L_LR"]
            h_lr = config["H_LR"]

        actor_ts = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=_make_tx(l_lr, config["L_MAX_GRAD_NORM"]),
        )
        critic_ts = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=_make_tx(l_lr, config["L_MAX_GRAD_NORM"]),
        )
        coord_ts = TrainState.create(
            apply_fn=coord.apply,
            params=coord_params,
            tx=_make_tx(h_lr, config["H_MAX_GRAD_NORM"]),
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
                world_state=world_state,      # pre-step (per-env, already computed above)
                team_skill_onehot=team_skill_onehot_actors,
                indi_skill_onehot=indi_skill_onehot_actors,
                env_reward=reward,           # (NUM_ENVS, n_agents)
                team_intri=team_intri_actors,
                indi_intri=indi_intri,
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
        # PPO + Discriminator updates (Milestone 4)
        # -----------------------------------------------------------------------

        # ---- Bootstrap: compute terminal values for GAE ----
        last_world_state = jnp.transpose(last_obs, (0, 2, 3, 1, 4)).reshape(NUM_ENVS, *ws_shape)

        # Re-run coordinator at terminal state to get team skill for low-level critic
        # and to get high-level bootstrap values.
        rng, _rng_boot = jax.random.split(rng)
        last_skill_actions, _, last_h_val = coord.apply(
            coord_ts.params, last_world_state, last_obs, _rng_boot,
            method=coord.get_actions,
        )
        # last_h_val: (NUM_ENVS, n_agents+1) — bootstrap for high-level GAE

        last_team_skill_idx = last_skill_actions[:, 0]                           # (NUM_ENVS,)
        last_team_skill_onehot_boot = jax.nn.one_hot(last_team_skill_idx, N_Z_TEAM)  # (NUM_ENVS, N_Z_TEAM)

        last_val_env, _ = critic.apply(
            critic_ts.params, last_world_state, last_team_skill_onehot_boot, rnn_critic
        )  # (NUM_ENVS,)
        last_val_actors = jnp.tile(last_val_env[None, :], (n_agents, 1)).reshape(NUM_ACTORS)
        # last_val_actors: (NUM_ACTORS,) — bootstrap for low-level GAE

        # ---- GAE helper ----
        def _compute_gae(traj_dones, traj_values, traj_rewards, last_val, gamma, gae_lambda):
            def _gae_step(carry, x):
                gae, next_val = carry
                done, value, reward = x
                delta = reward + gamma * next_val * (1.0 - done) - value
                gae = delta + gamma * gae_lambda * (1.0 - done) * gae
                return (gae, value), (gae, gae + value)

            _, (advantages, targets) = jax.lax.scan(
                _gae_step,
                (jnp.zeros_like(last_val), last_val),
                (traj_dones, traj_values, traj_rewards),
                reverse=True,
                unroll=16,
            )
            return advantages, targets

        # Low-level GAE: (NUM_STEPS, NUM_ACTORS)
        l_advantages, l_targets = _compute_gae(
            l_traj.done, l_traj.value, l_traj.reward,
            last_val_actors,
            config["L_GAMMA"], config["L_GAE_LAMBDA"],
        )

        # High-level GAE: broadcast done to (SKILL_STEPS, NUM_ENVS, n_agents+1)
        h_done_bc = jnp.tile(h_traj.done[:, :, None], (1, 1, n_agents + 1))
        h_advantages, h_targets = _compute_gae(
            h_done_bc, h_traj.value, h_traj.reward,
            last_h_val,
            config["H_GAMMA"], config["H_GAE_LAMBDA"],
        )
        # h_advantages: (SKILL_STEPS, NUM_ENVS, n_agents+1)

        # ---- Discriminator update ----
        # Reference: d_trainer.py discri_update() — discrete cross-entropy

        # Team discriminator data: per-env world_state + per-env team skill index
        # team_skill_onehot is agent-major (NUM_STEPS, NUM_ACTORS, N_Z_TEAM)
        # Reshape to (NUM_STEPS, n_agents, NUM_ENVS, N_Z_TEAM), take agent-0 slice for per-env
        ws_disc_flat = l_traj.world_state.reshape(config["NUM_STEPS"] * NUM_ENVS, *ws_shape)
        team_idx_flat = jnp.argmax(
            l_traj.team_skill_onehot.reshape(config["NUM_STEPS"], n_agents, NUM_ENVS, N_Z_TEAM)[:, 0, :, :],
            axis=-1,
        ).reshape(config["NUM_STEPS"] * NUM_ENVS)

        # Individual discriminator data: per-actor obs + team_skill + indi skill index
        obs_disc_flat = l_traj.obs.reshape(config["NUM_STEPS"] * NUM_ACTORS, *obs_shape)
        team_oh_disc_flat = l_traj.team_skill_onehot.reshape(config["NUM_STEPS"] * NUM_ACTORS, N_Z_TEAM)
        indi_idx_flat = jnp.argmax(
            l_traj.indi_skill_onehot.reshape(config["NUM_STEPS"] * NUM_ACTORS, N_Z_INDI),
            axis=-1,
        )

        def _discri_epoch(disc_states, unused):
            team_disc_ts_inner, indi_disc_ts_inner = disc_states

            # Team discriminator gradient step
            def _team_loss(params):
                logits = team_disc.apply(params, ws_disc_flat)
                return optax.softmax_cross_entropy_with_integer_labels(logits, team_idx_flat).mean()

            team_loss, team_grads = jax.value_and_grad(_team_loss)(team_disc_ts_inner.params)
            team_disc_ts_inner = team_disc_ts_inner.apply_gradients(grads=team_grads)

            # Individual discriminator gradient step
            def _indi_loss(params):
                logits = indi_disc.apply(params, obs_disc_flat, team_oh_disc_flat)
                return optax.softmax_cross_entropy_with_integer_labels(logits, indi_idx_flat).mean()

            indi_loss, indi_grads = jax.value_and_grad(_indi_loss)(indi_disc_ts_inner.params)
            indi_disc_ts_inner = indi_disc_ts_inner.apply_gradients(grads=indi_grads)

            return (team_disc_ts_inner, indi_disc_ts_inner), (team_loss, indi_loss)

        (team_disc_ts, indi_disc_ts), disc_losses = jax.lax.scan(
            _discri_epoch, (team_disc_ts, indi_disc_ts), None, config["D_EPOCH"]
        )
        # disc_losses: (D_EPOCH,) for team and indi respectively

        # ---- Low-level PPO update ----
        # Actor and critic have different batch sizes (NUM_ACTORS vs NUM_ENVS per step),
        # so they get separate minibatch loops inside the same epoch function.
        # Reference: r_mappo.py ppo_update()

        # Actor batch: per-actor, (NUM_STEPS * NUM_ACTORS) samples
        actor_batch = (
            l_traj.obs.reshape(config["NUM_STEPS"] * NUM_ACTORS, *obs_shape),
            l_traj.action.reshape(config["NUM_STEPS"] * NUM_ACTORS),
            l_traj.log_prob.reshape(config["NUM_STEPS"] * NUM_ACTORS),
            l_traj.rnn_state_actor.reshape(config["NUM_STEPS"] * NUM_ACTORS, HIDDEN_SIZE),
            l_traj.team_skill_onehot.reshape(config["NUM_STEPS"] * NUM_ACTORS, N_Z_TEAM),
            l_traj.indi_skill_onehot.reshape(config["NUM_STEPS"] * NUM_ACTORS, N_Z_INDI),
            l_advantages.reshape(config["NUM_STEPS"] * NUM_ACTORS),
        )

        # Critic batch: per-env, (NUM_STEPS * NUM_ENVS) samples
        # l_traj.value is agent-major (NUM_STEPS, NUM_ACTORS); reshape to (NUM_STEPS, n_agents, NUM_ENVS)
        # and take agent-0 slice — all agents in the same env share the same critic value
        vals_env = l_traj.value.reshape(config["NUM_STEPS"], n_agents, NUM_ENVS)[:, 0, :]
        tgts_env = l_targets.reshape(config["NUM_STEPS"], n_agents, NUM_ENVS)[:, 0, :]
        team_oh_env = l_traj.team_skill_onehot.reshape(
            config["NUM_STEPS"], n_agents, NUM_ENVS, N_Z_TEAM
        )[:, 0, :, :]  # (NUM_STEPS, NUM_ENVS, N_Z_TEAM)

        critic_batch = (
            l_traj.world_state.reshape(config["NUM_STEPS"] * NUM_ENVS, *ws_shape),
            l_traj.rnn_state_critic.reshape(config["NUM_STEPS"] * NUM_ENVS, HIDDEN_SIZE),
            team_oh_env.reshape(config["NUM_STEPS"] * NUM_ENVS, N_Z_TEAM),
            vals_env.reshape(config["NUM_STEPS"] * NUM_ENVS),
            tgts_env.reshape(config["NUM_STEPS"] * NUM_ENVS),
        )

        def _actor_update_minibatch(actor_ts_inner, batch):
            obs_mb, action_mb, old_lp_mb, rnn_mb, team_oh_mb, indi_oh_mb, adv_mb = batch

            def _actor_loss(params):
                pi, _ = actor.apply(params, obs_mb, team_oh_mb, indi_oh_mb, rnn_mb)
                lp = pi.log_prob(action_mb)
                entropy = pi.entropy().mean()
                ratio = jnp.exp(lp - old_lp_mb)
                adv_norm = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                loss_actor = -jnp.minimum(
                    ratio * adv_norm,
                    jnp.clip(ratio, 1 - config["L_CLIP_EPS"], 1 + config["L_CLIP_EPS"]) * adv_norm,
                ).mean()
                total = loss_actor - config["L_ENT_COEF"] * entropy
                return total, (loss_actor, entropy)

            grad_fn = jax.value_and_grad(_actor_loss, has_aux=True)
            (_, aux), grads = grad_fn(actor_ts_inner.params)
            actor_ts_inner = actor_ts_inner.apply_gradients(grads=grads)
            return actor_ts_inner, aux  # aux = (loss_actor, entropy) both scalars

        def _critic_update_minibatch(critic_ts_inner, batch):
            ws_mb, rnn_mb, team_oh_mb, old_val_mb, targets_mb = batch

            def _critic_loss(params):
                value, _ = critic.apply(params, ws_mb, team_oh_mb, rnn_mb)
                val_clipped = old_val_mb + jnp.clip(
                    value - old_val_mb, -config["L_CLIP_EPS"], config["L_CLIP_EPS"]
                )
                vf_loss = 0.5 * jnp.maximum(
                    (value - targets_mb) ** 2,
                    (val_clipped - targets_mb) ** 2,
                ).mean()
                return config["L_VF_COEF"] * vf_loss, vf_loss

            grad_fn = jax.value_and_grad(_critic_loss, has_aux=True)
            (_, vf_loss), grads = grad_fn(critic_ts_inner.params)
            critic_ts_inner = critic_ts_inner.apply_gradients(grads=grads)
            return critic_ts_inner, vf_loss  # vf_loss scalar

        def _low_update_epoch(low_state, unused):
            actor_ts_inner, critic_ts_inner, actor_b, critic_b, rng_inner = low_state
            rng_inner, rng_a, rng_c = jax.random.split(rng_inner, 3)

            # Shuffle and split actor data into minibatches
            n_a = config["NUM_STEPS"] * NUM_ACTORS
            perm_a = jax.random.permutation(rng_a, n_a)
            actor_shuf = jax.tree_util.tree_map(lambda x: jnp.take(x, perm_a, axis=0), actor_b)
            actor_mini = jax.tree_util.tree_map(
                lambda x: x.reshape((config["L_NUM_MINIBATCHES"], -1) + x.shape[1:]), actor_shuf
            )
            actor_ts_inner, actor_aux = jax.lax.scan(
                _actor_update_minibatch, actor_ts_inner, actor_mini
            )

            # Shuffle and split critic data into minibatches
            n_c = config["NUM_STEPS"] * NUM_ENVS
            perm_c = jax.random.permutation(rng_c, n_c)
            critic_shuf = jax.tree_util.tree_map(lambda x: jnp.take(x, perm_c, axis=0), critic_b)
            critic_mini = jax.tree_util.tree_map(
                lambda x: x.reshape((config["L_NUM_MINIBATCHES"], -1) + x.shape[1:]), critic_shuf
            )
            critic_ts_inner, critic_vf = jax.lax.scan(
                _critic_update_minibatch, critic_ts_inner, critic_mini
            )

            return (actor_ts_inner, critic_ts_inner, actor_b, critic_b, rng_inner), (actor_aux, critic_vf)

        rng, _rng_low = jax.random.split(rng)
        low_state, low_loss_info = jax.lax.scan(
            _low_update_epoch,
            (actor_ts, critic_ts, actor_batch, critic_batch, _rng_low),
            None,
            config["L_UPDATE_EPOCHS"],
        )
        actor_ts, critic_ts = low_state[0], low_state[1]

        # ---- High-level PPO update ----
        # H_NUM_MINIBATCHES=1 → full-batch gradient steps, H_UPDATE_EPOCHS epochs.
        # coordinator.evaluate() does teacher-forcing re-evaluation (no RNN).
        # Reference: same PPO clip + clipped value loss as low-level.

        h_ws_flat = h_traj.world_state.reshape(SKILL_STEPS * NUM_ENVS, *ws_shape)
        h_allobs_flat = h_traj.all_obs.reshape(SKILL_STEPS * NUM_ENVS, n_agents, *obs_shape)
        h_acts_flat = h_traj.skill_actions.reshape(SKILL_STEPS * NUM_ENVS, n_agents + 1)
        h_oldlp_flat = h_traj.log_prob.reshape(SKILL_STEPS * NUM_ENVS, n_agents + 1)
        h_oldval_flat = h_traj.value.reshape(SKILL_STEPS * NUM_ENVS, n_agents + 1)
        h_adv_flat = h_advantages.reshape(SKILL_STEPS * NUM_ENVS, n_agents + 1)
        h_tgt_flat = h_targets.reshape(SKILL_STEPS * NUM_ENVS, n_agents + 1)

        def _high_update_epoch(coord_ts_inner, unused):
            def _high_loss(params):
                log_probs, values, entropy = coord.apply(
                    params, h_ws_flat, h_allobs_flat, h_acts_flat,
                    method=coord.evaluate,
                )
                ratio = jnp.exp(log_probs - h_oldlp_flat)
                adv_norm = (h_adv_flat - h_adv_flat.mean()) / (h_adv_flat.std() + 1e-8)
                loss_actor = -jnp.minimum(
                    ratio * adv_norm,
                    jnp.clip(ratio, 1 - config["H_CLIP_EPS"], 1 + config["H_CLIP_EPS"]) * adv_norm,
                ).mean()
                val_clipped = h_oldval_flat + jnp.clip(
                    values - h_oldval_flat, -config["H_CLIP_EPS"], config["H_CLIP_EPS"]
                )
                vf_loss = 0.5 * jnp.maximum(
                    (values - h_tgt_flat) ** 2,
                    (val_clipped - h_tgt_flat) ** 2,
                ).mean()
                ent_loss = entropy.mean()
                total = loss_actor + config["H_VF_COEF"] * vf_loss - config["H_ENT_COEF"] * ent_loss
                return total, (loss_actor, vf_loss, ent_loss)

            grad_fn = jax.value_and_grad(_high_loss, has_aux=True)
            (total, aux), grads = grad_fn(coord_ts_inner.params)
            coord_ts_inner = coord_ts_inner.apply_gradients(grads=grads)
            return coord_ts_inner, (total, *aux)

        coord_ts, h_loss_info = jax.lax.scan(
            _high_update_epoch, coord_ts, None, config["H_UPDATE_EPOCHS"]
        )
        # h_loss_info: (total, loss_actor, vf_loss, ent_loss) each shape (H_UPDATE_EPOCHS,)

        # ---- Reassemble train_states ----
        train_states = (actor_ts, critic_ts, coord_ts, team_disc_ts, indi_disc_ts)

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

        # Intrinsic reward episode totals.
        # team_intri is the same value tiled across all agents in an env, so take agent-0
        # slice to avoid overcounting by n_agents. indi_intri is per-actor so sum over agents.
        # Shape: l_traj.team_intri / indi_intri = (NUM_STEPS, NUM_ACTORS), agent-major.
        _intri_team = l_traj.team_intri.reshape(config["NUM_STEPS"], n_agents, NUM_ENVS)
        _intri_indi  = l_traj.indi_intri.reshape(config["NUM_STEPS"], n_agents, NUM_ENVS)
        metric["rollout/team_intri_ep_total"] = _intri_team[:, 0, :].sum(axis=0).mean()
        metric["rollout/indi_intri_ep_total"]  = _intri_indi.sum(axis=(0, 1)).mean()

        # Training loss metrics (final epoch / final minibatch of each update)
        # disc_losses: tuple of (D_EPOCH,) arrays for team and indi
        metric["train/team_disc_loss"] = disc_losses[0][-1]
        metric["train/indi_disc_loss"] = disc_losses[1][-1]
        # low_loss_info: (actor_aux, critic_vf) where actor_aux = (loss_actor, entropy)
        # each element shape (L_UPDATE_EPOCHS, L_NUM_MINIBATCHES)
        metric["train/l_actor_loss"] = low_loss_info[0][0][-1, -1]
        metric["train/l_entropy"]    = low_loss_info[0][1][-1, -1]
        metric["train/l_value_loss"] = low_loss_info[1][-1, -1]
        # h_loss_info: (total, loss_actor, vf_loss, ent_loss) each shape (H_UPDATE_EPOCHS,)
        metric["train/h_total_loss"] = h_loss_info[0][-1]
        metric["train/h_actor_loss"] = h_loss_info[1][-1]
        metric["train/h_value_loss"] = h_loss_info[2][-1]
        metric["train/h_entropy"]    = h_loss_info[3][-1]

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
# Evaluate (hierarchical: coordinator assigns skills every SKILL_INTERVAL steps)
# ============================================================================

def evaluate(actor_params, coord_params, env, config, wandb_step: int, log_gif: bool = False):
    """Hierarchical eval: coordinator assigns team/individual skills every SKILL_INTERVAL steps.

    Mirrors MAPPO evaluate() structure: multi-episode loop, averaged metrics, GIF from
    first episode only. Uses NUM_STEPS per episode (not GIF_NUM_FRAMES).
    """
    HIDDEN_SIZE = config["HIDDEN_SIZE"]
    N_Z_TEAM = config["N_Z_TEAM"]
    N_Z_INDI = config["N_Z_INDI"]
    SKILL_INTERVAL = config["SKILL_INTERVAL"]
    n_agents = env.num_agents
    eval_num_episodes = config["EVAL_NUM_EPISODES"]

    # Observation and world-state shapes (cleanup: H=11, W=11, C=13, C_ws=91)
    obs_shape = env.observation_space()[0].shape   # (H, W, C)
    H, W, C = obs_shape
    C_ws = n_agents * C

    actor_net = SkillActor(
        action_dim=env.action_space().n,
        hidden_size=HIDDEN_SIZE,
        n_z_team=N_Z_TEAM,
        n_z_indi=N_Z_INDI,
        activation=config["ACTIVATION"],
    )
    coord_net = SkillCoordinator(
        n_agents=n_agents,
        n_z_team=N_Z_TEAM,
        n_z_indi=N_Z_INDI,
        n_block=config["N_BLOCK"],
        n_embd=config["N_EMBD"],
        n_head=config["N_HEAD"],
        activation=config["ACTIVATION"],
    )

    rng = jax.random.PRNGKey(0)

    raw_return_agents_sum = jnp.zeros((n_agents,), dtype=jnp.float32)
    raw_return_team_sum = 0.0
    raw_variance_sum = 0.0
    opt_tgt_return_team_sum = 0.0

    pics = []
    root_dir = "evaluation/cleanup"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for episode_idx in range(eval_num_episodes):
        rng, _rng_reset = jax.random.split(rng)
        obs, state = env.reset(_rng_reset)
        done = False

        # GRU state reset at episode start: (n_agents, HIDDEN_SIZE)
        rnn_state = jnp.zeros((n_agents, HIDDEN_SIZE))

        episode_raw_return_agents = jnp.zeros((n_agents,), dtype=jnp.float32)
        episode_return_team = 0.0
        episode_pics = []

        # Skill tensors — will be set on first coordinator call at step 0
        current_team_skill_batch = jnp.zeros((n_agents, N_Z_TEAM))
        current_indi_skill_batch = jnp.zeros((n_agents, N_Z_INDI))

        if log_gif and episode_idx == 0:
            episode_pics.append(env.render(state))

        for step in range(config["NUM_STEPS"]):
            # --- Coordinator: called at every SKILL_INTERVAL boundary ---
            if step % SKILL_INTERVAL == 0:
                obs_stack = jnp.stack([obs[a] for a in env.agents])    # (n_agents, H, W, C)
                # World state: transpose channel-major → spatial layout (matches training)
                # Training: (NUM_ENVS, n_agents, H, W, C) → transpose (0,2,3,1,4) → (NUM_ENVS, H, W, n_agents, C) → reshape (NUM_ENVS, H, W, C_ws)
                # Eval single-env: (n_agents, H, W, C) → transpose (1,2,0,3) → (H, W, n_agents, C) → reshape (H, W, C_ws)
                ws = obs_stack.transpose(1, 2, 0, 3).reshape(H, W, C_ws)
                world_state_eval = ws[None, :]              # (1, H, W, C_ws)
                all_obs_eval = obs_stack[None, :]           # (1, n_agents, H, W, C)

                rng, _rng_coord = jax.random.split(rng)
                skill_actions, _, _ = coord_net.apply(
                    coord_params,
                    world_state_eval, all_obs_eval, _rng_coord,
                    method=coord_net.get_actions,
                )
                # skill_actions: (1, n_agents+1) int32 — position 0 = team, 1..n = individual
                team_skill_idx = skill_actions[0, 0]        # scalar
                indi_skill_idx = skill_actions[0, 1:]       # (n_agents,)

                team_skill_onehot = jax.nn.one_hot(team_skill_idx, N_Z_TEAM)           # (N_Z_TEAM,)
                current_team_skill_batch = jnp.tile(
                    team_skill_onehot[None, :], (n_agents, 1)
                )                                                                        # (n_agents, N_Z_TEAM)
                current_indi_skill_batch = jax.nn.one_hot(indi_skill_idx, N_Z_INDI)    # (n_agents, N_Z_INDI)

            # --- Actor step ---
            obs_batch = jnp.stack([obs[a] for a in env.agents])  # (n_agents, H, W, C)
            pi, rnn_state = actor_net.apply(
                actor_params,
                obs_batch, current_team_skill_batch, current_indi_skill_batch, rnn_state,
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
            episode_raw_return_agents = episode_raw_return_agents + raw_step
            # opt_tgt: shared reward → mean (same value copied across agents);
            #          individual reward → sum (matches IPPO pattern)
            if config["ENV_KWARGS"]["shared_rewards"]:
                episode_return_team += float(reward.mean())
            else:
                episode_return_team += float(reward.sum())

            # GRU reset on episode done only — NOT at skill interval boundaries
            rnn_state = rnn_state * (1.0 - float(done))

            if log_gif and episode_idx == 0:
                episode_pics.append(env.render(state))

        # Per-episode accumulation
        raw_return_agents_sum += episode_raw_return_agents
        raw_return_team_sum += float(episode_raw_return_agents.sum())
        raw_variance_sum += float(jnp.var(episode_raw_return_agents))
        opt_tgt_return_team_sum += episode_return_team

        if log_gif and episode_idx == 0:
            pics = episode_pics

    # Average over episodes
    raw_return_agents = raw_return_agents_sum / eval_num_episodes
    raw_return_team = raw_return_team_sum / eval_num_episodes
    raw_variance = raw_variance_sum / eval_num_episodes
    return_team = opt_tgt_return_team_sum / eval_num_episodes

    eval_metrics = {}
    for i in range(n_agents):
        eval_metrics[f"eval/raw_return_agent{i}"] = float(raw_return_agents[i])
    eval_metrics["eval/raw_return_team"] = float(raw_return_team)
    eval_metrics["eval/raw_return_variance"] = float(raw_variance)
    eval_metrics["eval/opt_tgt_return_team"] = float(return_team)
    eval_metrics["eval/episodes_averaged"] = eval_num_episodes

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

        # Extract params for eval
        train_states = update_runner_state[0][0]
        actor_params = train_states[0].params
        coord_params = train_states[2].params

        update_step = int(update_runner_state[1])
        env_step = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

        evaluate(
            actor_params,
            coord_params,
            socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
            config=config,
            wandb_step=env_step,
            log_gif=True,
        )

    # Remainder chunk
    if remainder_updates > 0:
        update_runner_state, _ = remainder_jit(update_runner_state)

    # Final eval with GIF
    train_states = update_runner_state[0][0]
    actor_params = train_states[0].params
    coord_params = train_states[2].params
    update_step = int(update_runner_state[1])
    env_step = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

    evaluate(
        actor_params,
        coord_params,
        socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
        config=config,
        wandb_step=env_step,
        log_gif=True,
    )
    print("Finished training and evals")


# ============================================================================
# Hydra entry point
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="hmasd_cnn_cleanup_mini")
def main(config):
    print("Starting HMASD training with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    single_run(config)


if __name__ == "__main__":
    main()
