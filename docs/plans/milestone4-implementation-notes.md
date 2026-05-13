# Milestone 4 Implementation Notes

These notes carry forward everything discovered during Milestone 3 so the next session doesn't
need to re-derive context. Milestone 4 = GAE + PPO updates for all three learners (low-level,
high-level, discriminator) inside `_update_step`.

---

## Step 0: Add `world_state` to `LowTransition` (do this first)

`world_state` is NOT currently stored in `LowTransition`. The low-level critic needs it during
PPO re-evaluation. Do not try to reconstruct it from `l_traj.obs` — store it explicitly.

```python
class LowTransition(NamedTuple):
    global_done: jnp.ndarray        # (NUM_ACTORS,)
    done: jnp.ndarray               # (NUM_ACTORS,)
    action: jnp.ndarray             # (NUM_ACTORS,)
    value: jnp.ndarray              # (NUM_ACTORS,) — tiled from (NUM_ENVS,)
    reward: jnp.ndarray             # (NUM_ACTORS,) — combined intrinsic+extrinsic
    log_prob: jnp.ndarray           # (NUM_ACTORS,)
    obs: jnp.ndarray                # (NUM_ACTORS, H, W, C)
    rnn_state_actor: jnp.ndarray    # (NUM_ACTORS, HIDDEN_SIZE) — pre-step
    rnn_state_critic: jnp.ndarray   # (NUM_ENVS, HIDDEN_SIZE) — pre-step
    world_state: jnp.ndarray        # (NUM_ENVS, H, W, C_ws)  ← ADD THIS
    team_skill_onehot: jnp.ndarray  # (NUM_ACTORS, N_Z_TEAM)
    indi_skill_onehot: jnp.ndarray  # (NUM_ACTORS, N_Z_INDI)
    env_reward: jnp.ndarray         # (NUM_ENVS, n_agents) — raw env reward
    info: dict
```

In `_env_step`, `world_state` is already computed at the point where the critic runs (line ~265).
Just add it to the `LowTransition(...)` constructor call. No extra computation needed.

Also need `team_skill_onehot_envs` (per-env, `(NUM_ENVS, N_Z_TEAM)`) stored somewhere accessible
for discriminator update. It's currently in the inner carry — either pass it through or store it
in LowTransition. Easiest: add a `team_skill_onehot_envs` field `(NUM_ENVS, N_Z_TEAM)` or
derive it from `l_traj.team_skill_onehot` (reshape to `(NUM_STEPS, n_agents, NUM_ENVS, N_Z_TEAM)`
and take `[:, 0, :, :]` for agent-0 = per-env copy). The latter avoids a new field.

Similarly, `team_skill_idx` (int, per-env) and `indi_skill_idx` (int, per-actor) are needed for
discriminator CE loss. Easiest: derive from stored one-hots via `jnp.argmax`.

---

## Update Order Inside `_update_step`

After collecting the nested scan output, the three updates run sequentially:

1. Compute bootstrap values (no gradient, just forward passes)
2. Discriminator update (cross-entropy, `D_EPOCH` epochs)
3. Low-level PPO update (`L_UPDATE_EPOCHS` epochs)
4. High-level PPO update (`H_UPDATE_EPOCHS` epochs)

Reference: `base_runner.py compute()` + `overcooked_runner.py` train calls.
Order of 3/4 doesn't matter much — discriminator first means better intrinsic rewards
for the PPO loss computation, which matches the reference.

---

## Bootstrap (Step 1)

Must happen before any GAE. The outer scan leaves `last_obs`, `rnn_actor`, `rnn_critic` in
the runner state — these are the states AFTER the last rollout step.

```python
# Get world state at end of rollout
last_world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(NUM_ENVS, *ws_shape)
last_all_obs = last_obs  # (NUM_ENVS, n_agents, H, W, C)

# --- Low-level bootstrap ---
# Need skills for the terminal state — re-run coordinator (no grad needed here)
rng, _rng_coord = jax.random.split(rng)
last_skill_actions, _, _ = coord.apply(
    coord_ts.params, last_world_state, last_all_obs, _rng_coord,
    method=coord.get_actions
)
last_team_skill_idx = last_skill_actions[:, 0]                        # (NUM_ENVS,)
last_team_skill_onehot = jax.nn.one_hot(last_team_skill_idx, N_Z_TEAM)  # (NUM_ENVS, N_Z_TEAM)

last_val_env, _ = critic.apply(
    critic_ts.params, last_world_state, last_team_skill_onehot, rnn_critic
)  # (NUM_ENVS,)
last_val_actors = jnp.tile(last_val_env[None, :], (n_agents, 1)).reshape(NUM_ACTORS)  # (NUM_ACTORS,)

# --- High-level bootstrap ---
last_h_val = coord.apply(
    coord_ts.params, last_world_state, last_all_obs,
    method=coord.get_values
)  # (NUM_ENVS, n_agents+1)
```

Reference: `base_runner.py` `compute()` method — same pattern of re-running coordinator
to get skills, then running critic with those skills for bootstrap.

---

## GAE

Standard reverse-scan GAE. Write one helper and call it for both levels.

```python
def _compute_gae(traj_batch, last_val, gamma, gae_lambda):
    """
    traj_batch: NamedTuple with .done, .value, .reward — all (T, B, ...)
    last_val: (B, ...) bootstrap value
    Returns: advantages (T, B, ...), targets (T, B, ...)
    """
    def _gae_step(carry, transition):
        gae, next_val = carry
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + gamma * next_val * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, value), (gae, gae + value)

    _, (advantages, targets) = jax.lax.scan(
        _gae_step,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
    )
    return advantages, targets
```

**Low-level call:**
```python
# l_traj after reshape: fields are (NUM_STEPS, NUM_ACTORS) or (NUM_STEPS, NUM_ENVS, ...)
# done field to use: l_traj.done (NUM_STEPS, NUM_ACTORS) — per-agent done, correct for GAE mask
l_advantages, l_targets = _compute_gae(l_traj, last_val_actors,
                                        config["L_GAMMA"], config["L_GAE_LAMBDA"])
```

**High-level call:**
```python
# h_traj: fields are (SKILL_STEPS, NUM_ENVS, ...) or (SKILL_STEPS, NUM_ENVS, n_agents+1)
# h_traj.done: (SKILL_STEPS, NUM_ENVS) — episode done at end of each interval
# h_traj.value: (SKILL_STEPS, NUM_ENVS, n_agents+1)
# h_traj.reward: (SKILL_STEPS, NUM_ENVS, n_agents+1)

# GAE done must broadcast to match value/reward shape:
h_done_broadcast = jnp.tile(h_traj.done[:, :, None], (1, 1, n_agents + 1))
# Build a lightweight NamedTuple or pass fields directly:
h_advantages, h_targets = _compute_gae(
    NamedTuple_with(done=h_done_broadcast, value=h_traj.value, reward=h_traj.reward),
    last_h_val,
    config["H_GAMMA"], config["H_GAE_LAMBDA"]
)
# h_advantages: (SKILL_STEPS, NUM_ENVS, n_agents+1)
```

---

## Discriminator Update

Cross-entropy only — no PPO clip, no old/new log_prob ratio. Just standard supervised learning
on the stored transition data.

```python
def _discri_loss(params_team, params_indi, batch):
    # batch contains: world_state, obs, team_skill_onehot, team_skill_idx, indi_skill_idx, indi_skill_onehot
    team_logits = team_disc.apply(params_team, batch.world_state)   # (B, N_Z_TEAM)
    indi_logits = indi_disc.apply(params_indi, batch.obs, batch.team_skill_onehot)  # (B, N_Z_INDI)

    team_loss = optax.softmax_cross_entropy_with_integer_labels(
        team_logits, batch.team_skill_idx
    ).mean()
    indi_loss = optax.softmax_cross_entropy_with_integer_labels(
        indi_logits, batch.indi_skill_idx
    ).mean()
    return team_loss + indi_loss, (team_loss, indi_loss)
```

Config: `D_EPOCH=15, D_NUM_MINIBATCHES=1` — so effectively 15 full-batch gradient steps.

Data for discriminator comes from `l_traj`:
- `world_state`: `(NUM_STEPS, NUM_ENVS, H, W, C_ws)` — stored in LowTransition (Step 0)
- `obs`: `(NUM_STEPS, NUM_ACTORS, H, W, C)` — stored
- `team_skill_onehot`: `(NUM_STEPS, NUM_ACTORS, N_Z_TEAM)` → derive per-env copy for team_disc
- `team_skill_idx`: derive from `team_skill_onehot` via `jnp.argmax(l_traj.team_skill_onehot, -1)`
  → shape `(NUM_STEPS, NUM_ACTORS)` → take `[:, :NUM_ENVS]` for per-env (agent-0 slice)
- `indi_skill_idx`: `jnp.argmax(l_traj.indi_skill_onehot, -1)` → `(NUM_STEPS, NUM_ACTORS)`

Team discriminator operates on per-env world_state — flatten to `(NUM_STEPS*NUM_ENVS, H, W, C_ws)`.
Individual discriminator operates per-actor — flatten to `(NUM_STEPS*NUM_ACTORS, H, W, C)`.

---

## Low-Level PPO Update

Re-evaluates actor and critic using stored per-step RNN states (single GRU step per stored
transition, not full sequence unroll). This is the standard "stored-state recurrent PPO" approach
used in MAPPO cleanup — each stored transition is a self-contained forward pass.

```python
def _low_level_loss(actor_params, critic_params, batch, advantages, targets):
    # batch fields have shape (MINIBATCH_SIZE, ...)

    # Actor re-evaluation (single GRU step using stored pre-step RNN state)
    pi, _ = actor.apply(
        actor_params,
        batch.obs,                # (MB, H, W, C)
        batch.team_skill_onehot,  # (MB, N_Z_TEAM)
        batch.indi_skill_onehot,  # (MB, N_Z_INDI)
        batch.rnn_state_actor,    # (MB, HIDDEN_SIZE) — stored pre-step state
    )
    log_prob = pi.log_prob(batch.action)
    entropy = pi.entropy().mean()

    # PPO clip
    ratio = jnp.exp(log_prob - batch.log_prob)  # batch.log_prob = old log prob from rollout
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    clip_eps = config["L_CLIP_EPS"]
    loss_actor = -jnp.minimum(
        ratio * adv,
        jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    ).mean()

    # Critic re-evaluation
    value, _ = critic.apply(
        critic_params,
        batch.world_state,           # (MB, H, W, C_ws) — per-env, so MB = MINIBATCH_ENVS
        batch.team_skill_onehot_envs, # (MB, N_Z_TEAM) — per-env team skill
        batch.rnn_state_critic,       # (MB, HIDDEN_SIZE) — per-env stored state
    )
    # Clipped value loss
    value_clipped = batch.value + jnp.clip(value - batch.value, -clip_eps, clip_eps)
    vf_loss = jnp.maximum(
        (value - targets) ** 2,
        (value_clipped - targets) ** 2
    ).mean()

    total_loss = loss_actor + config["L_VF_COEF"] * vf_loss - config["L_ENT_COEF"] * entropy
    return total_loss, (loss_actor, vf_loss, entropy)
```

**Minibatch shape mismatch warning**: actor operates on `(NUM_ACTORS,)` = per-actor, critic
operates on `(NUM_ENVS,)` = per-env. These are different sizes per step. Two options:
- Option A: separate minibatches for actor (over NUM_ACTORS) and critic (over NUM_ENVS)
- Option B: tile value targets/advantages from per-env to per-actor (already done in `l_traj.value`
  which is tiled). Then critic minibatch is over NUM_ENVS while actor is over NUM_ACTORS,
  and you just sync by using `batch_size // n_agents` envs per actor minibatch.

Option B (tile and use the tiled value for actor loss, run critic separately) is cleaner.
Look at MAPPO cleanup's `_update_minibatch` for the exact pattern.

Config: `L_UPDATE_EPOCHS=2, L_NUM_MINIBATCHES=4`.

---

## High-Level PPO Update

Coordinator re-evaluation uses the `evaluate` method (teacher-forcing, single parallel forward).

```python
def _high_level_loss(coord_params, batch, advantages, targets):
    # batch: (MB, NUM_ENVS_per_MB, ...)
    log_probs, values, entropy = coord.apply(
        coord_params,
        batch.world_state,   # (MB, H, W, C_ws)
        batch.all_obs,       # (MB, n_agents, H, W, C)
        batch.skill_actions, # (MB, n_agents+1) int32
        method=coord.evaluate,
    )
    # log_probs, values, entropy: all (MB, n_agents+1)

    # PPO clip — operate on all n_agents+1 positions jointly
    ratio = jnp.exp(log_probs - batch.log_prob)  # (MB, n_agents+1)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    clip_eps = config["H_CLIP_EPS"]
    loss_actor = -jnp.minimum(
        ratio * adv,
        jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    ).mean()

    vf_loss = ((values - targets) ** 2).mean()
    ent_loss = entropy.mean()

    total = loss_actor + config["H_VF_COEF"] * vf_loss - config["H_ENT_COEF"] * ent_loss
    return total, (loss_actor, vf_loss, ent_loss)
```

No RNN in coordinator — no stored state needed. h_traj has everything.

Config: `H_UPDATE_EPOCHS=15, H_NUM_MINIBATCHES=1` — 15 full-batch gradient steps over
`(SKILL_STEPS * NUM_ENVS)` = ~`(750 * 32)` samples (with NUM_STEPS=1000, SKILL_INTERVAL=25).

---

## Key Shape Reference

After outer scan + reshape:

| Field | Shape | Notes |
|-------|-------|-------|
| `l_traj.obs` | `(NUM_STEPS, NUM_ACTORS, H, W, C)` | agent-major |
| `l_traj.world_state` | `(NUM_STEPS, NUM_ENVS, H, W, C_ws)` | per-env |
| `l_traj.rnn_state_actor` | `(NUM_STEPS, NUM_ACTORS, HIDDEN_SIZE)` | pre-step, agent-major |
| `l_traj.rnn_state_critic` | `(NUM_STEPS, NUM_ENVS, HIDDEN_SIZE)` | pre-step, per-env |
| `l_traj.value` | `(NUM_STEPS, NUM_ACTORS)` | tiled from per-env |
| `l_traj.reward` | `(NUM_STEPS, NUM_ACTORS)` | combined intrinsic+extrinsic |
| `l_traj.done` | `(NUM_STEPS, NUM_ACTORS)` | per-agent done |
| `l_traj.team_skill_onehot` | `(NUM_STEPS, NUM_ACTORS, N_Z_TEAM)` | agent-major |
| `l_traj.indi_skill_onehot` | `(NUM_STEPS, NUM_ACTORS, N_Z_INDI)` | agent-major |
| `l_traj.env_reward` | `(NUM_STEPS, NUM_ENVS, n_agents)` | raw env reward, env-major |
| `h_traj.world_state` | `(SKILL_STEPS, NUM_ENVS, H, W, C_ws)` | interval start state |
| `h_traj.all_obs` | `(SKILL_STEPS, NUM_ENVS, n_agents, H, W, C)` | interval start obs |
| `h_traj.skill_actions` | `(SKILL_STEPS, NUM_ENVS, n_agents+1)` | [Z, z1..z_n] |
| `h_traj.value` | `(SKILL_STEPS, NUM_ENVS, n_agents+1)` | coordinator value |
| `h_traj.log_prob` | `(SKILL_STEPS, NUM_ENVS, n_agents+1)` | coordinator log_prob |
| `h_traj.reward` | `(SKILL_STEPS, NUM_ENVS, n_agents+1)` | same value tiled n_agents+1 |
| `h_traj.done` | `(SKILL_STEPS, NUM_ENVS)` | episode done at interval end |

NUM_STEPS = SKILL_STEPS * SKILL_INTERVAL (e.g. 1000 = 40 * 25).

---

## Network API for PPO Updates

```python
# Low-level actor re-evaluation
pi, new_rnn = actor.apply(params, obs, team_skill_onehot, indi_skill_onehot, rnn_state_actor)
# → pi: distrax.Categorical, new_rnn: (B, HIDDEN_SIZE)
# log_prob = pi.log_prob(actions)  — shape (B,)
# entropy = pi.entropy()           — shape (B,)

# Low-level critic re-evaluation
value, new_rnn = critic.apply(params, world_state, team_skill_onehot_envs, rnn_state_critic)
# world_state: (B, H, W, C_ws), team_skill_onehot_envs: (B, N_Z_TEAM)
# → value: (B,), new_rnn: (B, HIDDEN_SIZE)

# High-level coordinator re-evaluation (teacher-forcing, no RNN)
log_probs, values, entropy = coord.apply(
    params, world_state, all_obs, skill_actions, method=coord.evaluate
)
# world_state: (B, H, W, C_ws), all_obs: (B, n_agents, H, W, C), skill_actions: (B, n_agents+1)
# → all (B, n_agents+1)

# High-level coordinator bootstrap (value only)
values = coord.apply(params, world_state, all_obs, method=coord.get_values)
# → (B, n_agents+1)

# Discriminators (feedforward, no RNN — just re-apply on stored obs/world_state)
team_logits = team_disc.apply(params, world_state)          # (B, N_Z_TEAM)
indi_logits = indi_disc.apply(params, obs, team_skill_onehot)  # (B, N_Z_INDI)
```

---

## Logging Additions for Milestone 4

Once updates are live, add to metric dict in `_update_step`:
```python
metric["train/l_actor_loss"] = ...
metric["train/l_value_loss"] = ...
metric["train/l_entropy"] = ...
metric["train/h_actor_loss"] = ...
metric["train/h_value_loss"] = ...
metric["train/h_entropy"] = ...
metric["train/team_disc_loss"] = ...
metric["train/indi_disc_loss"] = ...
```

---

## Reference Files

- `algorithms/HMASD/hmasd_networks.py` — all 5 network forward pass signatures (verified M3)
- `algorithms/MAPPO/mappo_cnn_cleanup.py` — `_update_minibatch`, GAE pattern, minibatch loop
- `NeurIPS2023_HMASD_code/hmasd/runner/shared/base_runner.py` — `compute()` method (bootstrap)
- `NeurIPS2023_HMASD_code/hmasd/runner/shared/overcooked_runner.py` — update call order
- `NeurIPS2023_HMASD_code/hmasd/algorithms/r_mappo/r_mappo.py` — low-level PPO loss details
- `NeurIPS2023_HMASD_code/hmasd/algorithms/r_mappo/algorithm/rMAPPOPolicy.py` — evaluate_actions signature
- `NeurIPS2023_HMASD_code/hmasd/algorithms/discriminator/d_trainer.py` — discriminator update loop
