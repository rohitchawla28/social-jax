# Milestone 3: HMASD Training Loop Skeleton + Basic Env Integration

## Context

Milestone 2 delivered `hmasd_networks.py` with all 5 network modules (SkillActor, SkillCritic, TeamDiscriminator, IndividualDiscriminator, SkillCoordinator) plus intrinsic reward helpers. Milestone 3 builds the training loop in `hmasd_cnn_cleanup.py` and config yaml. The goal is a nested-scan rollout collection that runs end-to-end with correct transition shapes. PPO updates are **stubs** (Milestone 4). Evaluate/WandB/single_run are **stubs** (Milestone 5).

## Files to Create

1. `algorithms/HMASD/config/hmasd_cnn_cleanup.yaml` — Hydra config
2. `algorithms/HMASD/hmasd_cnn_cleanup.py` — Training loop

## Design Decisions

### Memory: Don't tile world_state in LowTransition
With NUM_ENVS=32, tiling world_state to NUM_ACTORS (224) costs ~10 GB. Instead:
- Store `world_state` per-env `(NUM_ENVS, 11, 11, 91)` in LowTransition
- Store `obs` per-actor `(NUM_ACTORS, 11, 11, 13)` — reconstruct world_state from obs during PPO (Milestone 4)
- Store `value` per-actor `(NUM_ACTORS,)` — tiled from `(NUM_ENVS,)` critic output (cheap, just scalars)
- Store `rnn_state_critic` per-env `(NUM_ENVS, hidden)` — NOT tiled

Total memory per rollout: ~1.7 GB with NUM_ENVS=32, ~200 MB with NUM_ENVS=4 (for testing).

### Done convention: use current-step done (not last_done)
MAPPO stores `last_done` (previous step) in transition.done — this appears to be a bug (noted with a TODO). For correct GAE: `delta = r_t + γ * V(s_{t+1}) * (1 - d_t) - V(s_t)`, `d_t` must indicate whether transition t was terminal. Store current-step `done_batch` in `LowTransition.done`.

### Wrappers: LogWrapper only
`MAPPOWorldStateWrapper` is effectively a no-op (world_state lines commented out). Just use `LogWrapper` from `socialjax.wrappers.baselines`.

### Critic operates per-env, value tiled to per-actor
SkillCritic takes `(NUM_ENVS, H, W, C_ws)` → `(NUM_ENVS,)` value. Tile value to `(NUM_ACTORS,)` agent-major for transition storage and GAE alignment.

### Agent-major flattening (matching MAPPO)
Observations: `(NUM_ENVS, n_agents, H, W, C)` → transpose `(n_agents, NUM_ENVS, ...)` → reshape `(NUM_ACTORS, ...)`. All per-actor fields use this convention.

### env_reward stored env-major for high-level aggregation
`env_reward: (NUM_ENVS, n_agents)` stays env-major (direct from env.step) — only used for high-level reward sum, never enters PPO minibatching.

---

## Config: `algorithms/HMASD/config/hmasd_cnn_cleanup.yaml`

Copy structure from `mappo_cnn_cleanup.yaml`. Key values from the Milestone 1 plan:

```yaml
ENV_NAME: "clean_up"
ENV_KWARGS:
  num_agents: 7
  num_inner_steps: 1000
  shared_rewards: False
  cnn: True
  jit: True

TOTAL_TIMESTEPS: 3e8
NUM_ENVS: 32
NUM_STEPS: 1000
SKILL_INTERVAL: 25
SEED: 30
ACTIVATION: "relu"
GIF_NUM_FRAMES: 250

N_Z_TEAM: 3
N_Z_INDI: 3
LAMBDA_ENV: 100
LAMBDA_TEAM: 0.1
LAMBDA_INDI: 0.1

# Low-level PPO
L_LR: 5e-4
L_UPDATE_EPOCHS: 2
L_NUM_MINIBATCHES: 4
L_GAMMA: 0.99
L_GAE_LAMBDA: 0.95
L_CLIP_EPS: 0.2
L_ENT_COEF: 0.01
L_VF_COEF: 0.5
L_MAX_GRAD_NORM: 0.5

# High-level PPO (transformer)
H_LR: 5e-4
H_UPDATE_EPOCHS: 15
H_NUM_MINIBATCHES: 1
H_GAMMA: 0.99
H_GAE_LAMBDA: 0.95
H_CLIP_EPS: 0.2
H_ENT_COEF: 0.01
H_VF_COEF: 1.0
H_MAX_GRAD_NORM: 10.0

# Transformer
N_BLOCK: 1
N_EMBD: 64
N_HEAD: 1
HIDDEN_SIZE: 64

# Discriminator
D_TEAM_LR: 5e-4
D_INDI_LR: 5e-4
D_EPOCH: 15
D_NUM_MINIBATCHES: 1
D_MAX_GRAD_NORM: 10.0

# WandB
ENTITY: "rohit-chawla28"
PROJECT: "socialjax-ippo"
WANDB_GROUP: "HMASD-cleanup"
WANDB_MODE: "online"
```

---

## Training Loop: `algorithms/HMASD/hmasd_cnn_cleanup.py`

### Section 1: Imports (~20 lines)

```python
import jax, jax.numpy as jnp, flax.linen as nn, numpy as np, optax, distrax
from flax.training.train_state import TrainState
from typing import NamedTuple
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb, os, pickle
from PIL import Image
from pathlib import Path

from hmasd_networks import (
    SkillActor, SkillCritic, SkillCoordinator,
    TeamDiscriminator, IndividualDiscriminator,
    compute_team_intrinsic_reward, compute_indi_intrinsic_reward,
)
```

### Section 2: NamedTuples + Helpers (~40 lines)

```python
class LowTransition(NamedTuple):
    global_done: jnp.ndarray       # (NUM_ACTORS,)
    done: jnp.ndarray              # (NUM_ACTORS,) — current-step done for GAE
    action: jnp.ndarray            # (NUM_ACTORS,)
    value: jnp.ndarray             # (NUM_ACTORS,) — tiled from (NUM_ENVS,)
    reward: jnp.ndarray            # (NUM_ACTORS,) — combined low-level reward
    log_prob: jnp.ndarray          # (NUM_ACTORS,)
    obs: jnp.ndarray               # (NUM_ACTORS, 11, 11, 13)
    rnn_state_actor: jnp.ndarray   # (NUM_ACTORS, 64) — pre-step
    rnn_state_critic: jnp.ndarray  # (NUM_ENVS, 64) — pre-step, per-env
    team_skill_onehot: jnp.ndarray # (NUM_ACTORS, N_Z_TEAM)
    indi_skill_onehot: jnp.ndarray # (NUM_ACTORS, N_Z_INDI)
    env_reward: jnp.ndarray        # (NUM_ENVS, n_agents) — raw, env-major
    info: dict

class HighTransition(NamedTuple):
    done: jnp.ndarray           # (NUM_ENVS,) — done at interval end
    world_state: jnp.ndarray    # (NUM_ENVS, 11, 11, 91) — interval start
    all_obs: jnp.ndarray        # (NUM_ENVS, 7, 11, 11, 13) — interval start
    skill_actions: jnp.ndarray  # (NUM_ENVS, n_agents+1) — [Z, z1..z7] int
    value: jnp.ndarray          # (NUM_ENVS, n_agents+1) — coordinator values
    log_prob: jnp.ndarray       # (NUM_ENVS, n_agents+1)
    reward: jnp.ndarray         # (NUM_ENVS, n_agents+1) — summed env reward
```

`batchify`, `batchify_numpy`, `unbatchify` — copy verbatim from `mappo_cnn_cleanup.py` lines 134-149.

### Section 3: `make_train(config)` — top-level function

#### 3a: Env + Config derivations

```python
env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
config["SKILL_STEPS"] = config["NUM_STEPS"] // config["SKILL_INTERVAL"]  # 40
config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]

env = LogWrapper(env)
```

#### 3b: Network instantiation

5 networks, using config values for hyperparams. Key shapes for reference:
- `obs_shape = env.observation_space()[0].shape`  → `(11, 11, 13)`
- `ws_shape = (*obs_shape[:-1], obs_shape[-1] * env.num_agents)` → `(11, 11, 91)`

#### 3c: `_make_train_state(rng)` → 5 TrainStates

Init each network with dummy inputs of correct shape:
- **Actor**: `(1, *obs_shape)`, `(1, N_Z_TEAM)`, `(1, N_Z_INDI)`, `(1, HIDDEN_SIZE)`
- **Critic**: `(1, *ws_shape)`, `(1, N_Z_TEAM)`, `(1, HIDDEN_SIZE)`
- **Coordinator**: `(1, *ws_shape)`, `(1, n_agents, *obs_shape)`, `(1, n_agents+1)` int — init via `__call__` (evaluate mode)
- **Team disc**: `(1, *ws_shape)`
- **Indi disc**: `(1, *obs_shape)`, `(1, N_Z_TEAM)`

Optimizers: each uses `optax.chain(clip_by_global_norm(MAX_GRAD_NORM), adam(LR, eps=1e-5))` with the appropriate per-component LR and grad norm config keys. No LR annealing initially.

Return: `(actor_ts, critic_ts, coord_ts, team_disc_ts, indi_disc_ts), rng`

#### 3d: `init_runner_state(rng)` → runner_state tuple

```python
train_states, rng = _make_train_state(rng)
rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
obsv, env_state = jax.vmap(env.reset)(reset_rng)
# obsv: (NUM_ENVS, 7, 11, 11, 13)

rnn_actor = jnp.zeros((config["NUM_ACTORS"], config["HIDDEN_SIZE"]))
rnn_critic = jnp.zeros((config["NUM_ENVS"], config["HIDDEN_SIZE"]))
last_done = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.bool_)

rng, _rng = jax.random.split(rng)
return (train_states, env_state, obsv, last_done, rnn_actor, rnn_critic, _rng)
```

#### 3e: `_update_step(update_runner_state, unused)` — the main body

##### Unpack

```python
runner_state, update_steps = update_runner_state
train_states, env_state, last_obs, last_done, rnn_actor, rnn_critic, rng = runner_state
actor_ts, critic_ts, coord_ts, team_disc_ts, indi_disc_ts = train_states
```

##### `_env_step(inner_carry, unused)` — inner scan body (25 steps)

**Carry**: `(env_state, last_obs, rnn_actor, rnn_critic, team_skill_idx, indi_skill_idx_actors, team_skill_onehot_envs, team_skill_onehot_actors, indi_skill_onehot_actors, rng)`

Operations in order:

1. **Batchify obs** (agent-major):
   `obs_batch = jnp.transpose(last_obs, (1,0,2,3,4)).reshape(-1, *obs_shape)` → `(NUM_ACTORS, 11, 11, 13)`

2. **Actor forward**:
   `pi, new_rnn_actor = actor.apply(actor_ts.params, obs_batch, team_skill_onehot_actors, indi_skill_onehot_actors, rnn_actor)`
   Sample action + log_prob.

3. **World state** (per-env):
   `world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(NUM_ENVS, *ws_shape)` → `(NUM_ENVS, 11, 11, 91)`

4. **Critic forward** (per-env):
   `value, new_rnn_critic = critic.apply(critic_ts.params, world_state, team_skill_onehot_envs, rnn_critic)`
   `value` is `(NUM_ENVS,)`.

   Tile to per-actor (agent-major):
   ```python
   value_tiled = jnp.tile(value[None, :], (n_agents, 1)).reshape(NUM_ACTORS)
   ```

5. **Env step**:
   Unbatchify action → list → `jax.vmap(env.step)` → obsv, env_state, reward, done, info.

   Transpose info to agent-major:
   ```python
   info = jax.tree_util.tree_map(
       lambda x: jnp.transpose(x, (1, 0)).reshape(NUM_ACTORS), info
   )
   ```

6. **Intrinsic rewards** (on NEXT state, per paper Eq 4):
   - Team disc: `team_disc.apply(params, next_world_state)` → `(NUM_ENVS, N_Z_TEAM)` logits
     `team_intri = compute_team_intrinsic_reward(logits, team_skill_idx)` → `(NUM_ENVS,)`
     Tile to agent-major: `jnp.tile(team_intri[None,:], (n_agents,1)).reshape(NUM_ACTORS)`
   - Indi disc: `indi_disc.apply(params, next_obs_batch, team_skill_onehot_actors)` → `(NUM_ACTORS, N_Z_INDI)` logits
     `indi_intri = compute_indi_intrinsic_reward(logits, indi_skill_idx_actors)` → `(NUM_ACTORS,)`

7. **Combined reward**:
   ```python
   env_rew_actors = batchify_numpy(reward, env.agents, NUM_ACTORS).squeeze()
   combined = LAMBDA_ENV * env_rew_actors + LAMBDA_TEAM * team_intri_actors + LAMBDA_INDI * indi_intri
   ```

8. **GRU reset on done**:
   ```python
   ep_done = done["__all__"]  # (NUM_ENVS,)
   done_actors = jnp.tile(ep_done[None,:], (n_agents,1)).reshape(NUM_ACTORS)
   new_rnn_actor = new_rnn_actor * (1 - done_actors[:, None])
   new_rnn_critic = new_rnn_critic * (1 - ep_done[:, None])
   ```

9. **Build LowTransition** with pre-step RNN states, current-step done.

10. **Return** updated inner_carry + transition.

##### `_skill_interval(outer_carry, unused)` — outer scan body (40 steps)

**Carry**: `(env_state, last_obs, last_done, rnn_actor, rnn_critic, rng)`

Operations:

1. **World state + all_obs** for coordinator:
   ```python
   world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(NUM_ENVS, *ws_shape)
   all_obs = last_obs  # (NUM_ENVS, 7, 11, 11, 13)
   ```

2. **Coordinator forward** (autoregressive):
   ```python
   rng, _rng = jax.random.split(rng)
   skill_actions, skill_log_probs, skill_values = coord.apply(
       coord_ts.params, world_state, all_obs, _rng,
       method=coord.get_actions
   )
   # skill_actions: (NUM_ENVS, 8) int32 — [Z, z1..z7]
   team_skill_idx = skill_actions[:, 0]   # (NUM_ENVS,)
   indi_skill_idx = skill_actions[:, 1:]  # (NUM_ENVS, 7)
   ```

3. **Prepare skill tensors** for inner scan carry:
   - `team_skill_onehot_envs`: `one_hot(team_skill_idx, N_Z_TEAM)` → `(NUM_ENVS, N_Z_TEAM)`
   - `team_skill_onehot_actors`: tile to `(NUM_ACTORS, N_Z_TEAM)` agent-major
   - `indi_skill_onehot_actors`: `one_hot(indi_skill_idx, N_Z_INDI)` → `(NUM_ENVS, 7, N_Z_INDI)` → transpose(1,0,2) → reshape `(NUM_ACTORS, N_Z_INDI)`
   - `indi_skill_idx_actors`: `(NUM_ENVS, 7)` → transpose → reshape `(NUM_ACTORS,)`

4. **Inner scan** over SKILL_INTERVAL (25) steps → `inner_carry, l_traj`

5. **High-level reward**:
   ```python
   # l_traj.env_reward: (SKILL_INTERVAL, NUM_ENVS, n_agents)
   h_reward = l_traj.env_reward.sum(axis=0).mean(axis=-1)  # (NUM_ENVS,)
   h_reward = jnp.tile(h_reward[:, None], (1, n_agents + 1))  # (NUM_ENVS, 8)
   ```

6. **High-level done**: extract `last_done` from inner_carry (done at end of interval).

7. **Build HighTransition** with interval-start world_state/obs, coordinator outputs, aggregated reward.

8. **Return** `(outer_carry, (high_transition, l_traj))`

##### After outer scan

```python
outer_carry, (h_traj, l_traj) = jax.lax.scan(_skill_interval, outer_carry, None, SKILL_STEPS)

# Reshape l_traj: (SKILL_STEPS, SKILL_INTERVAL, ...) → (NUM_STEPS, ...)
l_traj = jax.tree_util.tree_map(
    lambda x: x.reshape((config["NUM_STEPS"],) + x.shape[2:]), l_traj
)
```

##### PPO stubs (Milestone 4)

```python
# TODO (Milestone 4): Low-level GAE + PPO for actor/critic
# TODO (Milestone 4): High-level GAE + PPO for coordinator
# TODO (Milestone 4): Discriminator cross-entropy updates
# train_states pass through unchanged for now
```

##### Metrics + callback

Reuse MAPPO's `_reduce_metric_dict` pattern for cleanup-specific metrics (raw_reward_individual, clean_action_info, cleaned_water). Add HMASD-specific:
- `rollout/combined_reward_mean` = l_traj.reward.mean()
- `rollout/env_reward_mean` = l_traj.env_reward.mean()

WandB callback via `jax.debug.callback`.

##### Return

```python
runner_state = (train_states, env_state, last_obs, last_done, rnn_actor, rnn_critic, rng)
return (runner_state, update_steps), metric
```

#### 3f: Chunk pattern

Identical to MAPPO: 10 eval chunks + remainder. Return `(init_runner_state, train_chunk, remainder_chunk, chunk_updates, remainder_updates)`.

### Section 4: `single_run(config)` — working stub

Minimal but functional. Follows MAPPO pattern:
1. Init WandB with config
2. Call `make_train(config)` → get init/chunk functions
3. JIT compile init + chunk
4. Run `init_runner_state(rng)` → runner_state
5. Loop 10 chunks, after each: extract actor params, call `evaluate()`, log rollout metrics
6. Remainder chunk if needed
7. Final eval with `log_gif=True`

This gives us real WandB metrics during testing even before PPO is implemented (rewards will be random policy level but the pipeline works end-to-end).

### Section 5: `evaluate()` — basic working version

Basic non-hierarchical eval for Milestone 3 (actor-only, random skills):
1. Reset env, sample random team_skill + indi_skills (fixed for episode)
2. Loop GIF_NUM_FRAMES steps: actor forward with skills + GRU → env step
3. Accumulate raw_return_agents, return_team
4. Log eval metrics to WandB
5. Optionally save + log GIF

Note: Full hierarchical eval (coordinator assigns skills every k steps) deferred to Milestone 5. The basic version here uses fixed random skills just to verify the actor works and produces frames.

### Section 6: Hydra entry point

```python
@hydra.main(version_base=None, config_path="config", config_name="hmasd_cnn_cleanup")
def main(config):
    single_run(config)
```

---

## Implementation Order

1. Create yaml config file
2. Write imports, NamedTuples, helpers (batchify/unbatchify)
3. Write `make_train` skeleton: env setup, config derivations, network instantiation
4. Write `_make_train_state` — init all 5 networks + create TrainStates
5. Write `init_runner_state` — env reset, initial zeros for RNN states
6. Write `_env_step` — actor/critic/discriminator forwards, env step, rewards, GRU reset
7. Write `_skill_interval` — coordinator forward, inner scan, high-level reward aggregation
8. Wire up `_update_step` — outer scan, reshape, metrics, callback
9. Write `_reduce_metric_dict` — cleanup + HMASD-specific metrics
10. Write chunk pattern (train_chunk, remainder_chunk)
11. Write `evaluate()` — basic actor-only eval with random skills + GIF
12. Write `single_run(config)` — WandB init, chunk loop, eval calls
13. Write Hydra entry point
14. Test: run with small config (NUM_ENVS=4, NUM_STEPS=50), verify shapes + no NaNs

---

## Verification

Run with small config (NUM_ENVS=4, NUM_STEPS=50, SKILL_INTERVAL=25 → SKILL_STEPS=2):

1. `init_runner_state` succeeds, returns correct tuple structure
2. 1 `_update_step` runs without errors
3. `l_traj` fields have correct shapes: `(50, 28, ...)` for per-actor, `(50, 4, ...)` for per-env
4. `h_traj` fields have correct shapes: `(2, 4, ...)` for per-env
5. No NaNs in any transition field
6. `team_skill_idx` values in `{0, 1, 2}`, `indi_skill_idx` values in `{0, 1, 2}`
7. Combined reward is finite
8. Intrinsic rewards in `(0, 1)` range (softmax probabilities)
9. GRU states reset when `done["__all__"]` fires

---

## Key Reference Files

- `algorithms/HMASD/hmasd_networks.py` — all network definitions + intrinsic reward helpers
- `algorithms/MAPPO/mappo_cnn_cleanup.py` — primary template (env setup, scan pattern, PPO, metrics, eval, single_run)
- `algorithms/MAPPO/config/mappo_cnn_cleanup.yaml` — config template
- `socialjax/wrappers/baselines.py` — LogWrapper (adds returned_episode_* to info)
- `docs/plans/hmasd-to-socialjax-plan.md` — Milestone 1 architecture + Milestone 2/3 implementation notes

## Risks

1. **Memory with NUM_ENVS=32**: ~1.7 GB for transitions. Should be OK on GPU, may be tight on M1 Mac. Test with NUM_ENVS=4 first.
2. **Inner scan carry size**: 10-element tuple. JAX handles this fine as a pytree, but watch for tracing slowdown during first JIT.
3. **JIT compile time**: Nested scan with transformer decoder (inner jax.lax.scan for autoregressive) means 3 levels of scan. First JIT may take 5-10 minutes. Use small config for testing.
4. **Agent-major tiling**: Most error-prone part. Use assertions during development: `assert obs_batch.shape == (NUM_ACTORS, *obs_shape)`.
