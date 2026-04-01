## Milestone 3 Implementation Notes

These notes pre-digest the patterns needed for `hmasd_cnn_cleanup.py` so a new session can implement the training loop skeleton.

### Network API Quick Reference (from hmasd_networks.py)

All signatures as implemented.

```python
# SkillActor (low-level, per-agent)
pi, new_rnn_state = actor.apply(params, obs, team_skill_onehot, indi_skill_onehot, rnn_state)
# obs: (NUM_ACTORS, H, W, C)
# team_skill_onehot: (NUM_ACTORS, N_Z_TEAM)
# indi_skill_onehot: (NUM_ACTORS, N_Z_INDI)
# rnn_state: (NUM_ACTORS, hidden_size)
# → pi: Categorical, new_rnn_state: (NUM_ACTORS, hidden_size)

# SkillCritic (low-level, centralized)
value, new_rnn_state = critic.apply(params, world_state, team_skill_onehot, rnn_state)
# world_state: (NUM_ENVS, H, W, C_ws)  — NOT batched by agent
# team_skill_onehot: (NUM_ENVS, N_Z_TEAM)
# rnn_state: (NUM_ENVS, hidden_size)
# → value: (NUM_ENVS,), new_rnn_state: (NUM_ENVS, hidden_size)

# TeamDiscriminator
logits = team_disc.apply(params, world_state)
# world_state: (NUM_ACTORS, H, W, C_ws)  — batched by agent for intrinsic reward
# → logits: (NUM_ACTORS, N_Z_TEAM)

# IndividualDiscriminator
logits = indi_disc.apply(params, obs, team_skill_onehot)
# obs: (NUM_ACTORS, H, W, C), team_skill_onehot: (NUM_ACTORS, N_Z_TEAM)
# → logits: (NUM_ACTORS, N_Z_INDI)

# Intrinsic reward helpers
team_reward = compute_team_intrinsic_reward(logits, team_skill_idx)
indi_reward = compute_indi_intrinsic_reward(logits, indi_skill_idx)
# Both: logits (NUM_ACTORS, n_z), idx (NUM_ACTORS,) int → reward (NUM_ACTORS,)
# Returns exp(log_softmax(logits)[idx])  — values in (0, 1)

# SkillCoordinator — three calling modes
actions, log_probs, values = coord.apply(params, world_state, all_obs, rng, method=coord.get_actions)
# world_state: (NUM_ENVS, H, W, C_ws), all_obs: (NUM_ENVS, n_agents, H, W, C)
# → actions: (NUM_ENVS, n_agents+1) int32
#   log_probs: (NUM_ENVS, n_agents+1)
#   values: (NUM_ENVS, n_agents+1)
#   actions[:, 0] = team_skill, actions[:, 1:] = indi_skills

log_probs, values, entropy = coord.apply(params, world_state, all_obs, actions, method=coord.evaluate)
# actions: (B, n_agents+1) int32  — for PPO update (teacher-forcing)
# → all (B, n_agents+1)

values = coord.apply(params, world_state, all_obs, method=coord.get_values)
# For bootstrap value at end of rollout
# → (B, n_agents+1)
```

### Observation Layout Conventions

**Critical**: In MAPPO (and HMASD should follow the same convention), `last_obs` is stored as a raw stacked array `(NUM_ENVS, n_agents, H, W, C)` — NOT as a dict. The env returns a stacked array after `jax.vmap(env.reset/step)`. IPPO's `batchify` (which expects a dict) is a different pattern — don't mix them.

```python
# last_obs (env-major stacked, as returned by vmap env):
#   shape: (NUM_ENVS, n_agents, H, W, C)

# For actor — flatten to agent-major (MAPPO line 251 pattern):
obs_batch = jnp.transpose(last_obs, (1,0,2,3,4)).reshape(-1, H, W, C)
# (NUM_ENVS, n_agents, H, W, C) → (n_agents, NUM_ENVS, H, W, C) → (NUM_ACTORS, H, W, C)

# For coordinator — env-major stacked (keep as-is, just reshape last two dims):
all_obs = last_obs  # (NUM_ENVS, n_agents, H, W, C) — exactly what coordinator expects
```

### World State Construction

**MAPPO lines 267-270 vs what HMASD needs** — these are different:

```python
# MAPPO line 267 (only this part is correct for HMASD):
world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(NUM_ENVS, H, W, -1)
# last_obs: (NUM_ENVS, n_agents, H, W, C)
# transpose (0,2,3,1,4): (NUM_ENVS, H, W, n_agents, C)
# reshape: (NUM_ENVS, H, W, n_agents*C)  e.g. (NUM_ENVS, 11, 11, 91)

# MAPPO lines 268-270 (DO NOT use for HMASD):
# world_state = jnp.expand_dims(world_state, axis=0)
# world_state = jnp.tile(world_state, (env.num_agents, 1, 1, 1, 1))
# world_state = jnp.reshape(world_state, (-1, *(world_state.shape[2:])))
# ↑ MAPPO tiles to (NUM_ACTORS, H, W, C_ws) because its critic is per-agent.
# HMASD critic and coordinator are per-env — stop at (NUM_ENVS, H, W, C_ws).
```

So for HMASD, world state construction is just line 267 and that's it. Use `(NUM_ENVS, H, W, C_ws)` for coordinator, critic, and team discriminator directly.

### Skill Broadcasting Pattern

The coordinator outputs `(NUM_ENVS, n_agents+1)`, but actor/critic/discriminators need `(NUM_ACTORS, ...)`:

```python
# team_skill: (NUM_ENVS,) int  [from actions[:, 0]]
# indi_skills: (NUM_ENVS, n_agents) int  [from actions[:, 1:]]

# For actor/critic: broadcast team_skill to each agent
team_skill_onehot = jax.nn.one_hot(team_skill, N_Z_TEAM)      # (NUM_ENVS, N_Z_TEAM)
team_skill_actors = jnp.repeat(team_skill_onehot[:, None, :], n_agents, axis=1)  # (NUM_ENVS, n_agents, N_Z_TEAM)
team_skill_actors = team_skill_actors.reshape(NUM_ACTORS, N_Z_TEAM)  # (NUM_ACTORS, N_Z_TEAM)

# For actor: per-agent individual skills
indi_skill_onehot = jax.nn.one_hot(indi_skills, N_Z_INDI)     # (NUM_ENVS, n_agents, N_Z_INDI)
indi_skill_actors = indi_skill_onehot.reshape(NUM_ACTORS, N_Z_INDI)  # (NUM_ACTORS, N_Z_INDI)

# For team discriminator: also needs team_skill per-actor (same broadcast)
# world_state for team discriminator: tile world_state to (NUM_ACTORS, H, W, C_ws)
world_state_tiled = jnp.repeat(world_state[:, None, ...], n_agents, axis=1)
world_state_actors = world_state_tiled.reshape(NUM_ACTORS, H, W, C_ws)
# team_skill_idx for reward: (NUM_ACTORS,)
team_skill_idx = jnp.repeat(team_skill, n_agents)             # (NUM_ACTORS,)
# indi_skill_idx for reward: (NUM_ACTORS,)
indi_skill_idx = indi_skills.reshape(NUM_ACTORS)               # (NUM_ACTORS,)
```

### GRU State Shapes and Reset

Two separate GRU states with different shapes — do not confuse them:

```python
# Actor: per-agent GRU state
rnn_state_actor: shape (NUM_ACTORS, hidden_size)  # NUM_ACTORS = NUM_ENVS * n_agents

# Critic: per-env GRU state (centralized critic sees world state, not per-agent)
rnn_state_critic: shape (NUM_ENVS, hidden_size)

# Reset on episode done (done comes from env step, shape (NUM_ENVS,)):
# Actor reset — broadcast done to each agent
done_actors = jnp.repeat(done, n_agents)[:, None]             # (NUM_ACTORS, 1)
rnn_state_actor = rnn_state_actor * (1.0 - done_actors)

# Critic reset — direct broadcast
rnn_state_critic = rnn_state_critic * (1.0 - done[:, None])   # (NUM_ENVS, hidden_size)
```

### Carry Structures (scan)

```python
# runner_state (persists across _update_step calls):
runner_state = (
    train_states,       # dict of 5 TrainState objects (see below)
    env_state,          # SocialJax VecEnv state
    last_obs,           # dict {agent: (NUM_ENVS, H, W, C)}
    rnn_state_actor,    # (NUM_ACTORS, hidden_size)
    rnn_state_critic,   # (NUM_ENVS, hidden_size)
    update_step,        # scalar int
    rng,                # JAX PRNG key
)

# outer_carry (per skill interval, passed into _skill_interval):
# Extract from runner_state, keep as flat tuple for scan compatibility:
outer_carry = (env_state, last_obs, rnn_state_actor, rnn_state_critic, rng)

# inner_carry (per env step, passed into _env_step):
inner_carry = (env_state, last_obs, rnn_state_actor, rnn_state_critic,
               team_skill,    # (NUM_ENVS,) int — fixed for this interval
               indi_skills,   # (NUM_ENVS, n_agents) int — fixed for this interval
               rng)
```

### High-Level Reward and Done Aggregation

After the inner scan completes for one skill interval:

```python
# low_transitions: LowTransition with fields of shape (SKILL_INTERVAL, NUM_ACTORS, ...)
# or (SKILL_INTERVAL, NUM_ENVS, ...) for per-env fields

# High-level reward: sum env rewards across interval, average across agents
env_rewards = low_transitions.reward  # (SKILL_INTERVAL, NUM_ACTORS)
env_rewards_shaped = env_rewards.reshape(SKILL_INTERVAL, NUM_ENVS, n_agents)
h_reward_per_env = env_rewards_shaped.sum(axis=0).mean(axis=-1)  # (NUM_ENVS,)
# Broadcast to n_agents+1 positions for high-level PPO
h_reward = jnp.tile(h_reward_per_env[:, None], (1, n_agents + 1))  # (NUM_ENVS, n_agents+1)

# High-level done: use the last done from the inner scan (end of interval)
# done from LowTransition: (SKILL_INTERVAL, NUM_ACTORS)
last_done = low_transitions.done[-1]  # (NUM_ACTORS,)
# Reduce to per-env (any agent done → env done):
h_done = last_done.reshape(NUM_ENVS, n_agents).any(axis=-1)  # (NUM_ENVS,)
# Broadcast to n_agents+1 for HighTransition:
h_done_broadcast = jnp.tile(h_done[:, None], (1, n_agents + 1))  # (NUM_ENVS, n_agents+1)
```

### init_runner_state Structure

Five `TrainState` objects — match the naming used in `_skill_interval`:

```python
from flax.training.train_state import TrainState
import optax

# Standard pattern from IPPO/MAPPO:
actor_state = TrainState.create(
    apply_fn=actor_network.apply,
    params=actor_network.init(rng_a, dummy_obs, dummy_team_onehot, dummy_indi_onehot, dummy_rnn_actor),
    tx=optax.chain(optax.clip_by_global_norm(config["L_MAX_GRAD_NORM"]),
                   optax.adam(config["L_LR"], eps=1e-5)),
)
critic_state = TrainState.create(...)  # L_LR, L_MAX_GRAD_NORM
coord_state = TrainState.create(...)   # H_LR, H_MAX_GRAD_NORM
team_disc_state = TrainState.create(...)  # D_TEAM_LR, D_MAX_GRAD_NORM
indi_disc_state = TrainState.create(...)  # D_INDI_LR, D_MAX_GRAD_NORM

train_states = (actor_state, critic_state, coord_state, team_disc_state, indi_disc_state)
# Use a tuple (not dict) for scan compatibility — JAX can trace tuples as pytrees

# Initial RNN states (zeros):
rnn_state_actor = jnp.zeros((NUM_ACTORS, config["HIDDEN_SIZE"]))
rnn_state_critic = jnp.zeros((NUM_ENVS, config["HIDDEN_SIZE"]))
```

### LowTransition — Reward Field is Combined Reward

The `reward` field in `LowTransition` should store the **combined intrinsic+extrinsic reward** (already weighted by lambdas), not raw env reward. Store raw env reward separately in `info` or as a separate field if you need it for the high-level reward aggregation and for logging. This is important: the high-level reward sums over `env_reward_raw`, not the combined reward.

```python
env_reward = ...  # raw from env step, per env: (NUM_ENVS,) → batchify to (NUM_ACTORS,)
team_intri = compute_team_intrinsic_reward(td_logits, team_skill_idx)   # (NUM_ACTORS,)
indi_intri = compute_indi_intrinsic_reward(id_logits, indi_skill_idx)   # (NUM_ACTORS,)

combined_reward = (config["LAMBDA_ENV"] * env_reward_actors
                   + config["LAMBDA_TEAM"] * team_intri
                   + config["LAMBDA_INDI"] * indi_intri)

# For high-level: use env_reward (NOT combined_reward) — see High-Level Reward section above
```

### Critical Reference Lines

- `algorithms/IPPO/ippo_cnn_cleanup.py`: `batchify`/`unbatchify` definitions, `make_train`, `init_runner_state`, scan pattern, `_env_step` structure
- `algorithms/MAPPO/mappo_cnn_cleanup.py:267` — world state construction (exact transpose+reshape)
- `algorithms/MAPPO/mappo_cnn_cleanup.py`: centralized critic call pattern, `Transition` with `world_state` field
- `NeurIPS2023_HMASD_code/hmasd/runner/shared/overcooked_runner.py` — if unclear about skill assignment timing or reward aggregation order
