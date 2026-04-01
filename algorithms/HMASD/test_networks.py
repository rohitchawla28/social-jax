"""Verification script for HMASD network definitions (Milestone 2)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from hmasd_networks import (
    CNN, SkillActor, SkillCritic,
    TeamDiscriminator, IndividualDiscriminator,
    SkillCoordinator,
    compute_team_intrinsic_reward, compute_indi_intrinsic_reward,
)

rng = jax.random.PRNGKey(0)
B, n_agents = 4, 7
H, W, C = 11, 11, 13
C_ws = C * n_agents  # 91

print("=" * 60)
print("HMASD Network Verification")
print("=" * 60)

# --- Test CNN ---
print("\n[1] CNN...")
cnn = CNN()
dummy_obs = jnp.zeros((B, H, W, C))
rng, _rng = jax.random.split(rng)
params = cnn.init(_rng, dummy_obs)
out = cnn.apply(params, dummy_obs)
assert out.shape == (B, 64), f"CNN output shape: {out.shape}"
assert not jnp.any(jnp.isnan(out)), "CNN has NaNs"
print(f"  OK: output shape {out.shape}, no NaNs")

# --- Test SkillActor ---
print("\n[2] SkillActor...")
actor = SkillActor(action_dim=9, hidden_size=64, n_z_team=3, n_z_indi=3)
dummy_team = jnp.zeros((B, 3))
dummy_indi = jnp.zeros((B, 3))
dummy_rnn = jnp.zeros((B, 64))
rng, _rng = jax.random.split(rng)
params = actor.init(_rng, dummy_obs, dummy_team, dummy_indi, dummy_rnn)
pi, new_rnn = actor.apply(params, dummy_obs, dummy_team, dummy_indi, dummy_rnn)
assert new_rnn.shape == (B, 64), f"Actor RNN state: {new_rnn.shape}"
rng, _rng = jax.random.split(rng)
action = pi.sample(seed=_rng)
assert action.shape == (B,), f"Actor action: {action.shape}"
log_prob = pi.log_prob(action)
assert log_prob.shape == (B,), f"Actor log_prob: {log_prob.shape}"
entropy = pi.entropy()
assert entropy.shape == (B,), f"Actor entropy: {entropy.shape}"
assert not jnp.any(jnp.isnan(new_rnn)), "Actor RNN has NaNs"
assert not jnp.any(jnp.isnan(log_prob)), "Actor log_prob has NaNs"
print(f"  OK: action {action.shape}, log_prob {log_prob.shape}, rnn {new_rnn.shape}, no NaNs")

# --- Test SkillCritic ---
print("\n[3] SkillCritic...")
critic = SkillCritic(hidden_size=64, n_z_team=3)
dummy_ws = jnp.zeros((B, H, W, C_ws))
rng, _rng = jax.random.split(rng)
params = critic.init(_rng, dummy_ws, dummy_team, dummy_rnn)
val, new_rnn_c = critic.apply(params, dummy_ws, dummy_team, dummy_rnn)
assert val.shape == (B,), f"Critic value: {val.shape}"
assert new_rnn_c.shape == (B, 64), f"Critic RNN state: {new_rnn_c.shape}"
assert not jnp.any(jnp.isnan(val)), "Critic value has NaNs"
print(f"  OK: value {val.shape}, rnn {new_rnn_c.shape}, no NaNs")

# --- Test TeamDiscriminator ---
print("\n[4] TeamDiscriminator...")
team_disc = TeamDiscriminator(n_z_team=3)
rng, _rng = jax.random.split(rng)
params = team_disc.init(_rng, dummy_ws)
logits = team_disc.apply(params, dummy_ws)
assert logits.shape == (B, 3), f"TeamDiscri logits: {logits.shape}"
assert not jnp.any(jnp.isnan(logits)), "TeamDiscri has NaNs"
# Test intrinsic reward
team_idx = jnp.array([0, 1, 2, 0])
reward = compute_team_intrinsic_reward(logits, team_idx)
assert reward.shape == (B,), f"Team reward: {reward.shape}"
assert not jnp.any(jnp.isnan(reward)), "Team reward has NaNs"
print(f"  OK: logits {logits.shape}, reward {reward.shape}, no NaNs")

# --- Test IndividualDiscriminator ---
print("\n[5] IndividualDiscriminator...")
indi_disc = IndividualDiscriminator(n_z_indi=3, n_z_team=3)
rng, _rng = jax.random.split(rng)
params = indi_disc.init(_rng, dummy_obs, dummy_team)
logits = indi_disc.apply(params, dummy_obs, dummy_team)
assert logits.shape == (B, 3), f"IndiDiscri logits: {logits.shape}"
assert not jnp.any(jnp.isnan(logits)), "IndiDiscri has NaNs"
indi_idx = jnp.array([0, 1, 2, 0])
reward = compute_indi_intrinsic_reward(logits, indi_idx)
assert reward.shape == (B,), f"Indi reward: {reward.shape}"
assert not jnp.any(jnp.isnan(reward)), "Indi reward has NaNs"
print(f"  OK: logits {logits.shape}, reward {reward.shape}, no NaNs")

# --- Test SkillCoordinator ---
print("\n[6] SkillCoordinator...")
coord = SkillCoordinator(
    n_agents=n_agents, n_z_team=3, n_z_indi=3,
    n_block=1, n_embd=64, n_head=1,
)
dummy_all_obs = jnp.zeros((B, n_agents, H, W, C))
dummy_actions_init = jnp.zeros((B, n_agents + 1), dtype=jnp.int32)
rng, _rng = jax.random.split(rng)
params = coord.init(_rng, dummy_ws, dummy_all_obs, dummy_actions_init)
print("  Init OK")

# Test get_values
values = coord.apply(params, dummy_ws, dummy_all_obs, method=coord.get_values)
assert values.shape == (B, n_agents + 1), f"Coordinator values: {values.shape}"
assert not jnp.any(jnp.isnan(values)), "Coordinator values have NaNs"
print(f"  get_values OK: {values.shape}, no NaNs")

# Test get_actions (autoregressive)
rng, _rng = jax.random.split(rng)
actions, log_probs, values = coord.apply(params, dummy_ws, dummy_all_obs, _rng, method=coord.get_actions)
assert actions.shape == (B, n_agents + 1), f"Coord actions: {actions.shape}"
assert log_probs.shape == (B, n_agents + 1), f"Coord log_probs: {log_probs.shape}"
assert values.shape == (B, n_agents + 1), f"Coord values: {values.shape}"
assert actions.dtype == jnp.int32, f"Coord actions dtype: {actions.dtype}"
assert not jnp.any(jnp.isnan(log_probs)), "Coord log_probs have NaNs"
assert not jnp.any(jnp.isnan(values)), "Coord values have NaNs"
team_skill = actions[:, 0]
indi_skills = actions[:, 1:]
assert team_skill.shape == (B,), f"team_skill: {team_skill.shape}"
assert indi_skills.shape == (B, n_agents), f"indi_skills: {indi_skills.shape}"
# Check actions are in valid range
assert jnp.all(actions[:, 0] < 3), f"Team skill out of range: {actions[:, 0]}"
assert jnp.all(actions[:, 1:] < 3), f"Indi skills out of range"
print(f"  get_actions OK: actions {actions.shape}, log_probs {log_probs.shape}, values {values.shape}")
print(f"    team_skill range: [0, {int(actions[:, 0].max())}], indi_skills range: [0, {int(actions[:, 1:].max())}]")

# Test evaluate (teacher forcing)
log_probs_e, values_e, entropy_e = coord.apply(params, dummy_ws, dummy_all_obs, actions, method=coord.evaluate)
assert log_probs_e.shape == (B, n_agents + 1), f"Eval log_probs: {log_probs_e.shape}"
assert values_e.shape == (B, n_agents + 1), f"Eval values: {values_e.shape}"
assert entropy_e.shape == (B, n_agents + 1), f"Eval entropy: {entropy_e.shape}"
assert not jnp.any(jnp.isnan(log_probs_e)), "Eval log_probs have NaNs"
assert not jnp.any(jnp.isnan(entropy_e)), "Eval entropy has NaNs"
print(f"  evaluate OK: log_probs {log_probs_e.shape}, values {values_e.shape}, entropy {entropy_e.shape}")

# Verify log_probs are consistent between get_actions and evaluate
# (They should match because evaluate re-computes with same actions)
lp_diff = jnp.abs(log_probs - log_probs_e).max()
print(f"  log_prob consistency: max diff = {float(lp_diff):.6f}")
if lp_diff > 1e-4:
    print(f"  WARNING: log_probs differ by {float(lp_diff):.6f} (expected < 1e-4)")

# --- Test with different n_z_team != n_z_indi ---
print("\n[7] SkillCoordinator with n_z_team=2, n_z_indi=5 (masking test)...")
coord2 = SkillCoordinator(
    n_agents=n_agents, n_z_team=2, n_z_indi=5,
    n_block=1, n_embd=64, n_head=1,
)
rng, _rng = jax.random.split(rng)
params2 = coord2.init(_rng, dummy_ws, dummy_all_obs, dummy_actions_init)
rng, _rng = jax.random.split(rng)
actions2, lp2, v2 = coord2.apply(params2, dummy_ws, dummy_all_obs, _rng, method=coord2.get_actions)
assert jnp.all(actions2[:, 0] < 2), f"Team skill should be < 2, got max={int(actions2[:, 0].max())}"
assert jnp.all(actions2[:, 1:] < 5), f"Indi skill should be < 5, got max={int(actions2[:, 1:].max())}"
print(f"  OK: team_skill range [0,{int(actions2[:, 0].max())}] (max 1), indi range [0,{int(actions2[:, 1:].max())}] (max 4)")

# Evaluate consistency
lp2_e, v2_e, ent2_e = coord2.apply(params2, dummy_ws, dummy_all_obs, actions2, method=coord2.evaluate)
lp2_diff = jnp.abs(lp2 - lp2_e).max()
print(f"  log_prob consistency: max diff = {float(lp2_diff):.6f}")

# --- Parameter counts ---
print("\n[8] Parameter counts...")
def count_params(params):
    return sum(x.size for x in jax.tree.leaves(params))

# Re-init for counting
rng, _rng = jax.random.split(rng)
actor_params = actor.init(_rng, dummy_obs, dummy_team, dummy_indi, dummy_rnn)
rng, _rng = jax.random.split(rng)
critic_params = critic.init(_rng, dummy_ws, dummy_team, dummy_rnn)
rng, _rng = jax.random.split(rng)
td_params = team_disc.init(_rng, dummy_ws)
rng, _rng = jax.random.split(rng)
id_params = indi_disc.init(_rng, dummy_obs, dummy_team)

print(f"  SkillActor:              {count_params(actor_params):>10,} params")
print(f"  SkillCritic:             {count_params(critic_params):>10,} params")
print(f"  TeamDiscriminator:       {count_params(td_params):>10,} params")
print(f"  IndividualDiscriminator: {count_params(id_params):>10,} params")
print(f"  SkillCoordinator:        {count_params(params):>10,} params")
total = count_params(actor_params) + count_params(critic_params) + count_params(td_params) + count_params(id_params) + count_params(params)
print(f"  Total:                   {total:>10,} params")

print("\n" + "=" * 60)
print("All verification checks PASSED!")
print("=" * 60)
