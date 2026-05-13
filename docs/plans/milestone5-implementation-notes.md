# Milestone 5 Implementation Notes

These notes carry forward everything discovered during the M4 session so the next session
doesn't need to re-derive context. Milestone 5 = hierarchical eval(), single_run() fixups,
EVAL_NUM_EPISODES config key.

---

## Current State of the File

`algorithms/HMASD/hmasd_cnn_cleanup.py` after M4 is complete through the PPO updates.
The following are the remaining issues:

- `evaluate()` (line ~814): **M3 stub** — actor-only, single episode, random fixed skills,
  uses `GIF_NUM_FRAMES` as step count. Must be fully replaced.
- `single_run()` (line ~928): structurally correct but has two bugs:
  1. The 10 chunk evals call `log_gif=False` — should be `log_gif=True` (matching IPPO)
  2. Only extracts `actor_params` — needs to also extract `coord_params`
- Config `hmasd_cnn_cleanup.yaml`: missing `EVAL_NUM_EPISODES` key.

---

## Step 0: Add EVAL_NUM_EPISODES to config

In `algorithms/HMASD/config/hmasd_cnn_cleanup.yaml`, add:
```yaml
EVAL_NUM_EPISODES: 10
```

---

## Step 1: Replace evaluate() — full signature and structure

### Signature change

Old:
```python
def evaluate(actor_params, env, save_path, config, wandb_step: int, log_gif: bool = False):
```

New (add `coord_params`, drop `save_path` which was never used):
```python
def evaluate(actor_params, coord_params, env, config, wandb_step: int, log_gif: bool = False):
```

### Network instantiation inside evaluate()

```python
n_agents = env.num_agents
obs_shape = env.observation_space()[0].shape      # (H, W, C)
ws_shape = (*obs_shape[:-1], obs_shape[-1] * n_agents)  # (H, W, C*n_agents)

actor_net = SkillActor(
    action_dim=env.action_space().n,
    hidden_size=config["HIDDEN_SIZE"],
    n_z_team=config["N_Z_TEAM"],
    n_z_indi=config["N_Z_INDI"],
    activation=config["ACTIVATION"],
)
coord_net = SkillCoordinator(
    n_agents=n_agents,
    n_z_team=config["N_Z_TEAM"],
    n_z_indi=config["N_Z_INDI"],
    n_block=config["N_BLOCK"],
    n_embd=config["N_EMBD"],
    n_head=config["N_HEAD"],
    activation=config["ACTIVATION"],
)
```

### World state construction for single env

In training, world_state is built as:
```python
# last_obs: (NUM_ENVS, n_agents, H, W, C)
world_state = jnp.transpose(last_obs, (0, 2, 3, 1, 4)).reshape(NUM_ENVS, *ws_shape)
```

For eval (single env, obs is a dict `{str(agent): (H, W, C)}`):
```python
obs_stack = jnp.stack([obs[a] for a in env.agents])        # (n_agents, H, W, C)
world_state_eval = jnp.transpose(obs_stack, (1, 2, 0, 3)).reshape(*ws_shape)  # (H, W, C*n_agents)
ws_batch     = world_state_eval[None, :]      # (1, H, W, C*n_agents)  — add batch dim
all_obs_batch = obs_stack[None, :]            # (1, n_agents, H, W, C) — add batch dim
```

`coord_net.get_actions` expects `(B, H, W, C_ws)` and `(B, n_agents, H, W, C)` with `B=1` here.

### Coordinator call — every SKILL_INTERVAL steps

The coordinator is called at the **start** of each skill interval, i.e. at steps 0, 25, 50, ... (i.e. `step % SKILL_INTERVAL == 0`). This matches the training outer scan which assigns skills before the inner scan starts.

```python
if step % SKILL_INTERVAL == 0:
    obs_stack = jnp.stack([obs[a] for a in env.agents])     # recompute from current obs
    ws_batch = jnp.transpose(obs_stack, (1, 2, 0, 3)).reshape(*ws_shape)[None, :]
    all_obs_batch = obs_stack[None, :]

    rng, _rng_coord = jax.random.split(rng)
    skill_actions, _, _ = coord_net.apply(
        coord_params, ws_batch, all_obs_batch, _rng_coord,
        method=coord_net.get_actions,
    )
    # skill_actions: (1, n_agents+1) — [Z, z1..z_n]
    team_skill_idx   = skill_actions[0, 0]       # scalar
    indi_skill_idx   = skill_actions[0, 1:]      # (n_agents,)

    team_skill_oh = jax.nn.one_hot(team_skill_idx, config["N_Z_TEAM"])           # (N_Z_TEAM,)
    team_skill_batch = jnp.tile(team_skill_oh[None, :], (n_agents, 1))           # (n_agents, N_Z_TEAM)
    indi_skill_batch = jax.nn.one_hot(indi_skill_idx, config["N_Z_INDI"])         # (n_agents, N_Z_INDI)

# Then unconditionally run actor with current skills:
obs_stack = jnp.stack([obs[a] for a in env.agents])
pi, rnn_state = actor_net.apply(actor_params, obs_stack, team_skill_batch, indi_skill_batch, rnn_state)
```

**Important: obs_stack is computed twice on skill interval boundaries** (once for coord, once
for actor). This is fine. You could cache it but it's a cheap operation.

**Do NOT reset rnn_state at skill interval boundaries.** GRU state only resets on episode done.
In training, `new_rnn_actor = new_rnn_actor * (1.0 - done_batch[:, None])` — only episode done
triggers a reset. The GRU carries temporal state across skill intervals within an episode.

### Multi-episode loop structure (matches IPPO/MAPPO)

```python
eval_num_episodes = config["EVAL_NUM_EPISODES"]

raw_return_agents_sum = jnp.zeros((n_agents,), dtype=jnp.float32)
raw_return_team_sum   = 0.0
raw_variance_sum      = 0.0
opt_tgt_return_sum    = 0.0
pics = []
root_dir = "evaluation/cleanup"
Path(root_dir + "/state_pics").mkdir(parents=True, exist_ok=True)

for episode_idx in range(eval_num_episodes):
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    rnn_state = jnp.zeros((n_agents, config["HIDDEN_SIZE"]))  # reset GRU per episode

    episode_raw_return_agents = jnp.zeros((n_agents,), dtype=jnp.float32)
    episode_return_team = 0.0
    episode_pics = []

    if log_gif and episode_idx == 0:
        episode_pics.append(env.render(state))  # NO np.array() needed — render() returns onp.ndarray

    for step in range(config["NUM_STEPS"]):  # use NUM_STEPS, not GIF_NUM_FRAMES
        if step % SKILL_INTERVAL == 0:
            ...coordinator call...

        obs_stack = jnp.stack([obs[a] for a in env.agents])
        pi, rnn_state = actor_net.apply(actor_params, obs_stack, team_skill_batch, indi_skill_batch, rnn_state)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)

        env_act = {k: v.squeeze() for k, v in unbatchify(actions, env.agents, 1, n_agents).items()}
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done_all = done["__all__"]

        raw_step = info["raw_reward_individual"]          # (n_agents,)
        episode_raw_return_agents += raw_step
        episode_return_team += float(reward.mean())       # see note below on shared_rewards

        rnn_state = rnn_state * (1.0 - float(done_all))  # reset GRU on episode done

        if log_gif and episode_idx == 0:
            episode_pics.append(env.render(state))

    raw_return_agents_sum += episode_raw_return_agents
    raw_return_team_sum   += float(episode_raw_return_agents.sum())
    raw_variance_sum      += float(jnp.var(episode_raw_return_agents))
    opt_tgt_return_sum    += episode_return_team

    if log_gif and episode_idx == 0:
        pics = episode_pics

# Average and log
raw_return_agents = raw_return_agents_sum / eval_num_episodes
raw_return_team   = raw_return_team_sum   / eval_num_episodes
raw_variance      = raw_variance_sum      / eval_num_episodes
return_team       = opt_tgt_return_sum    / eval_num_episodes

eval_metrics = {}
for i in range(n_agents):
    eval_metrics[f"eval/raw_return_agent{i}"] = float(raw_return_agents[i])
eval_metrics["eval/raw_return_team"]      = float(raw_return_team)
eval_metrics["eval/raw_return_variance"]  = float(raw_variance)
eval_metrics["eval/opt_tgt_return_team"]  = float(return_team)
eval_metrics["eval/episodes_averaged"]    = eval_num_episodes
wandb.log(eval_metrics, step=int(wandb_step))
```

**Note on `reward.mean()` vs `reward.sum()` for `shared_rewards=False`:**
MAPPO cleanup always uses `reward.mean()` for `opt_tgt_return_team`; IPPO uses
`sum(reward)` when `shared_rewards=False`. HMASD config has `shared_rewards: False`.
Use `reward.mean()` for consistency with MAPPO (HMASD is closer in structure to MAPPO).
This is a pre-existing inconsistency across baselines — don't add a branch.

### GIF saving

```python
if log_gif and pics:
    new_pics = [Image.fromarray(img) for img in pics]   # NO np.array() needed
    gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_frames-{len(new_pics)}.gif"
    new_pics[0].save(gif_path, format="GIF", save_all=True, optimize=False,
                     append_images=new_pics[1:], duration=200, loop=0)
    print("Logging GIF to WandB")
    wandb.log({"eval/episode_gif": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")},
              step=int(wandb_step))
```

**Why no `np.array(img)`**: The cleanup env's `render()` is typed `-> onp.ndarray` and built
entirely with `onp.zeros`, `onp.uint8`, plain numpy. `Image.fromarray` accepts it directly.
IPPO's `np.array(img)` is harmless but redundant; don't copy it. MAPPO doesn't use it.

---

## Step 2: Fix single_run()

Two changes:

**1. log_gif=True for all 10 chunk evals** (line ~966–973):

Old:
```python
evaluate(actor_params, ..., log_gif=False)
```
New:
```python
evaluate(actor_params, coord_params, ..., log_gif=True)
```
Remove the separate "Final eval with GIF" block — it becomes redundant since all 10 evals
already log GIFs. Or keep it as an 11th eval after the remainder chunk, matching IPPO
(which does `log_gif=True` for all 10 + 1 final = 11 total). This matches the IPPO pattern exactly.

**2. Extract coord_params alongside actor_params** (line ~960–961):

Old:
```python
train_states = update_runner_state[0][0]
actor_params = train_states[0].params
```

New:
```python
train_states = update_runner_state[0][0]
# train_states = (actor_ts, critic_ts, coord_ts, team_disc_ts, indi_disc_ts)
actor_params = train_states[0].params
coord_params = train_states[2].params
```

Apply the same extraction in the final eval block (line ~980–982).

---

## runner_state / update_runner_state structure reference

```
update_runner_state = (runner_state, update_steps)
update_runner_state[0]    = runner_state
update_runner_state[0][0] = train_states = (actor_ts, critic_ts, coord_ts, team_disc_ts, indi_disc_ts)
update_runner_state[0][1] = env_state
update_runner_state[0][2] = last_obs
update_runner_state[0][3] = last_done
update_runner_state[0][4] = rnn_actor
update_runner_state[0][5] = rnn_critic
update_runner_state[0][6] = rng
update_runner_state[1]    = update_steps (int counter)
```

`train_states[0]` = actor_ts → `.params` for actor
`train_states[2]` = coord_ts → `.params` for coordinator

---

## Env in eval — raw env, not LogWrapper

`single_run()` passes `socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])` (no LogWrapper).
The raw cleanup env natively provides `info["raw_reward_individual"]` (shape `(n_agents,)`).
`reward` from raw env step is also an array (not a dict) — `reward.mean()` works.

---

## SKILL_INTERVAL in evaluate()

`SKILL_INTERVAL = config["SKILL_INTERVAL"]` — add this at the top of `evaluate()` alongside the
other config extractions. It's already in the config (set in `make_train` as a derived key, but
also defined in the yaml).

---

## Summary of all changes for M5

| Location | Change |
|---|---|
| `hmasd_cnn_cleanup.yaml` | Add `EVAL_NUM_EPISODES: 10` |
| `evaluate()` signature | Add `coord_params`, drop `save_path` |
| `evaluate()` body | Full replace: multi-episode loop, coord every SKILL_INTERVAL, avg metrics |
| `single_run()` loop | `log_gif=True` for all 10 chunk evals |
| `single_run()` param extract | Add `coord_params = train_states[2].params` in both places |

No new files needed. No changes to training loop, networks, or config structure.

---

## Verification checks

1. Import succeeds after changes
2. `evaluate()` runs for 2 episodes with `NUM_STEPS=50, SKILL_INTERVAL=5`:
   - No shape errors
   - All eval metrics are finite scalars
   - GIF file is created and has frames
3. `single_run()` for 1 chunk (tiny config) logs eval metrics to WandB at each eval step
4. `eval/episodes_averaged` appears in WandB logs with value 10
