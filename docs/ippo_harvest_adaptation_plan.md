# IPPO Harvest Common — Adaptation Plan

## Context
Adapting `ippo_cnn_harvest_common.py` to match the research metric/eval setup already applied to `ippo_cnn_cleanup.py`. Goal is correctness and accurate WandB metrics. The structure, metric names, and `_reduce_metric_dict` logic should mirror `ippo_cnn_cleanup.py` exactly.

`harvest_open.py` now exposes `raw_reward_individual` in the info dict (line 1386/1438), identical to cleanup's convention — so `_reduce_metric_dict` can be copied verbatim from cleanup with zero key-name changes.

---

## Critical Files
- **Target:** [algorithms/IPPO/ippo_cnn_harvest_common.py](algorithms/IPPO/ippo_cnn_harvest_common.py)
- **Config:** [algorithms/IPPO/config/ippo_cnn_harvest_common.yaml](algorithms/IPPO/config/ippo_cnn_harvest_common.yaml)
- **Reference (mirror exactly):** [algorithms/IPPO/ippo_cnn_cleanup.py](algorithms/IPPO/ippo_cnn_cleanup.py)

---

## Phase 1 (Changes #1–5): Core training loop fixes

### Change 1 — Comment out hardcoded sys.path (line 5)
```python
# sys.path.append('/home/shuqing/SocialJax')
```

---

### Change 2 — Fix info agent-major transpose bug (line 304, PARAMETER_SHARING block)

Current (incorrect):
```python
info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
```

Replace with (mirrors cleanup lines 308–317):
```python
# info from vmap env.step is env-major (NUM_ENVS, num_agents);
# transpose to agent-major (num_agents, NUM_ENVS) before flattening
# to match obs/reward/done actor axis layout
info = jax.tree_util.tree_map(
    lambda x: jnp.transpose(x, (1, 0)).reshape((config["NUM_ACTORS"],)),
    info,
)
```

No remapping needed — `raw_reward_individual` is now directly in the harvest info dict.

---

### Change 3 — Refactor `_loss_fn` and `_update_minbatch` for structured train metrics

In `_loss_fn`: add `value_mean = value.mean()` and include in aux return (mirrors cleanup lines 435–442):
```python
return total_loss, (value_loss, loss_actor, entropy, value_mean)
```

In `_update_minbatch`: unpack new aux, compute grad norm, return `train_metrics` dict (mirrors cleanup lines 445–463):
```python
(total_loss, (value_loss, loss_actor, entropy, value_mean)), grads = grad_fn(
    train_state.params, traj_batch, advantages, targets, network_used
)
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
```

---

### Change 4 — Wire train metrics into rollout metric dict

After the epoch scan (PARAMETER_SHARING path), replace the current block with (mirrors cleanup lines 509–516):
```python
train_state = update_state[0]
metric = traj_batch.info
train_metric = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
metric = {**metric, **train_metric}
rng = update_state[-1]
```

Remove the old naive `metric = jax.tree_map(lambda x: x.mean(), metric)` call.

---

### Change 5 — Add `_reduce_metric_dict` and use it

Copy `_reduce_metric_dict` verbatim from cleanup (cleanup lines 543–615). No modifications needed — `raw_reward_individual` is the same key, and the `_sum_episode_total("clean_action_info")` / `_sum_episode_total("cleaned_water")` calls are silent no-ops for harvest (keys absent from `m`, function returns early).

Other harvest-specific info keys (`AppleCount_info`, `original_rewards`, `shaped_rewards`) fall through to the default `out[k] = v.mean()` path automatically.

Replace the metric logging block (mirrors cleanup lines 617–627):
```python
if config["PARAMETER_SHARING"]:
    metric = _reduce_metric_dict(metric)
else:
    metric = [_reduce_metric_dict(m_i) for m_i in metric]
    metric = metric[0]

metric["update_step"] = update_step
metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
jax.debug.callback(callback, metric)
```

Update `callback` to log with step (mirrors cleanup line 534):
```python
def callback(metric):
    wandb.log(metric, step=metric["env_step"])
```

---

## Phase 2 (Changes #6–9): Chunked eval loop + evaluate() refactor

*(To be implemented after Phase 1 is verified.)*

### Change 6 — Refactor `make_train` to return chunked runner state
Extract `init_runner_state` and return `init_runner_state, train_chunk, remainder_chunk, chunk_updates, remainder_updates` (mirrors cleanup lines 246–659 structure).

### Change 7 — Refactor `single_run` to use 10-eval chunk loop
Replace `jax.vmap(train_jit)(rngs)` with init → 10× chunk_jit + eval loop + remainder + final eval. Add `group=config["WANDB_GROUP"]` to `wandb.init`. Remove checkpoint save/load.

### Change 8 — Refactor `evaluate` function
Signature: `evaluate(params, env, save_path, config, wandb_step: int, log_gif: bool = False)`.
- Loop `config["EVAL_NUM_EPISODES"]` episodes × `config["NUM_STEPS"]` steps
- Use `info["original_rewards"]` for per-agent raw return tracking (eval uses bare env directly)
- Log `eval/raw_return_agent{i}`, `eval/raw_return_team`, `eval/raw_return_variance`, `eval/opt_tgt_return_team`, `eval/episodes_averaged`
- GIF only when `log_gif=True`, under `eval/episode_gif`

### Change 9 — Config updates (`ippo_cnn_harvest_common.yaml`)
- Add `EVAL_NUM_EPISODES: 10`
- Add `WANDB_GROUP: "IPPO-individual-harvest-common"`
- Change `TUNE: True` → `TUNE: False`

---

## Verification (Phase 1)
Run with a tiny test config (e.g. `NUM_ENVS=4, NUM_STEPS=16, TOTAL_TIMESTEPS=1024`):
1. No JAX shape errors at runtime
2. WandB shows `rollout/raw_ep_return_agent*`, `rollout/raw_ep_return_team`, `rollout/raw_ep_return_variance` at each update step
3. WandB shows `train/total_loss`, `train/grad_norm`, `train/value_mean`, etc.
4. `AppleCount_info`, `original_rewards`, `shaped_rewards` appear as mean scalars via default path
5. Episode keys (`returned_episode_returns` etc.) appear correctly via `v[-1].mean()`
