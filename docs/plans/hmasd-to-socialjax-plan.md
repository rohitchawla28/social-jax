# HMASD JAX Implementation Plan for SocialJax

## Context

We're porting HMASD (Hierarchical Multi-Agent Skill Discovery, NeurIPS 2023) from a PyTorch reference implementation (`NeurIPS2023_HMASD_code/`) into JAX/Flax, targeting SocialJax environments. The goal is an apples-to-apples comparable baseline alongside existing IPPO and MAPPO implementations. Starting with the **cleanup** environment (7 agents, 9 actions, 1000-step episodes, CNN observations).

HMASD is a two-level hierarchical algorithm:
- **High-level (Skill Coordinator)**: Transformer that assigns a team skill Z and per-agent individual skills z^i every k timesteps
- **Low-level (Skill Discoverer)**: Actor-critic with GRU that executes primitive actions conditioned on assigned skills
- **Discriminators**: Team discriminator q_D(Z|s) and individual discriminator q_d(z^i|o^i,Z) provide intrinsic rewards
- Both levels optimized with PPO; discriminators trained with supervised cross-entropy

## File Structure

```
algorithms/HMASD/
    hmasd_networks.py          # All Flax nn.Module definitions (~400 lines)
    hmasd_cnn_cleanup.py       # Training loop, eval, logging (~800 lines)
    config/
        hmasd_cnn_cleanup.yaml # Hydra config
```

Split rationale: HMASD has 5+ networks and 3 PPO update loops — a single file would exceed 1500 lines. Networks in a separate file are reusable across environments (harvest_open, coins later). Training file stays focused on the loop, eval, logging.

## Key Design Decisions

### 1. Keep RNNs (faithful to paper)
The reference uses GRU in the low-level actor and critic (NOT in the transformer or discriminators). Within a skill interval of k=25 steps, the agent keeps the same skill embedding — the GRU provides temporal reasoning for multi-step skill execution (e.g., sequential actions to complete a task). This is important for faithful reproduction.

RNN locations in HMASD:
- **Low-level Actor**: MLP → GRU → action_head (memory for skill execution)
- **Low-level Critic**: MLP → GRU → value_head (tracks skill execution progress)
- **Discriminators**: Feedforward only (use_recurrent_discri=0 in reference defaults)
- **Transformer Coordinator**: No RNN (operates on single timestep snapshots)

JAX implications: GRU hidden states must be carried through the inner `jax.lax.scan` and reset at episode boundaries. Hidden states also need to be stored in transitions for PPO sequence chunking.

### 2. CNN backbone everywhere
All networks use the same 3-layer CNN as IPPO cleanup (32-32-32 filters, 5x5/3x3/3x3, orthogonal init, relu). Skill embeddings concatenated to the 64-dim CNN feature vector (post-flatten, pre-head). This avoids changing CNN input dimensions and matches the reference's `skill_last_layer=1` mode.

### 3. World state = stacked agent observations
Following MAPPO cleanup: world_state concatenates all 7 agent observations along channels → shape `(11, 11, 13*7)=(11, 11, 91)`. Used for high-level critic and team discriminator.

### 4. Nested scan structure
- Outer `jax.lax.scan`: iterates over `NUM_STEPS // SKILL_INTERVAL` skill intervals (1000/25=40)
- Inner `jax.lax.scan`: iterates over `SKILL_INTERVAL` env steps per interval (25)
GRU hidden states are carried through inner scan, reset on episode done.

### 5. Full autoregressive transformer
Implement the complete transformer skill coordinator with autoregressive decoding via `jax.lax.scan` over n_agent+1 steps. Faithful to the paper's sequential skill assignment.

### 6. Overcooked defaults for hyperparameters
SKILL_INTERVAL=25, N_Z_TEAM=3, N_Z_INDI=3, λ_e=100, λ_D=0.1, λ_d=0.1. Can be tuned later.

### 7. NUM_ENVS = 32
HMASD has ~4x per-step computation vs IPPO. Start with 32, adjust if memory issues.

### 8. RNG key handling (same as IPPO/MAPPO)
Follow the identical `jax.random.split` threading pattern used in IPPO/MAPPO:
- `rng` is part of the `runner_state` tuple, threaded through every function
- Pattern: `rng, _rng = jax.random.split(rng)` → pass `_rng` to consumer, keep `rng` for next split
- Used in: `init_runner_state` (network init, env reset), `_skill_interval` (transformer autoregressive sampling), `_env_step` (low-level actor sampling, env step), PPO updates (minibatch shuffling)
- No deviation from the existing IPPO/MAPPO approach

### 9. No explicit buffer classes (jax.lax.scan replaces them)
The HMASD reference uses explicit buffer objects (`HSharedReplayBuffer`, `LSharedReplayBuffer`, `StateSkillDataset`) because PyTorch requires manual storage. In JAX, `jax.lax.scan` replaces all of these — scan automatically stacks returned NamedTuples into pytree arrays along a time axis. This is exactly how IPPO/MAPPO already work (no buffer class, just Transition NamedTuples).

Mapping:
- **`l_shared_buffer`** → `LowTransition` NamedTuple returned from inner scan. Shape `(SKILL_INTERVAL, NUM_ENVS*num_agents, ...)`. Fields: `obs, action, value, reward, log_prob, done, rnn_state_actor, rnn_state_critic, team_skill, indi_skill, info`
- **`h_shared_buffer`** → `HighTransition` NamedTuple returned from outer scan. Shape `(SKILL_STEPS, NUM_ENVS, ...)`. Fields: `world_state, obs, team_skill, indi_skills, h_value, h_log_prob, h_reward, done`
- **`state_skill_dataset`** → extract `(world_state, team_skill)` and `(obs, team_skill, indi_skill)` directly from the stacked low transitions for discriminator cross-entropy training. No separate dataset class needed.

After the nested scan completes, the full trajectory is already materialized as stacked arrays — ready for GAE computation, PPO minibatching, and discriminator training, just like IPPO's `traj_batch`.

## Network Architectures (hmasd_networks.py)

### CNN (shared backbone, identical to IPPO cleanup)
`Conv(32, 5x5) → ReLU → Conv(32, 3x3) → ReLU → Conv(32, 3x3) → ReLU → Flatten → Dense(64) → ReLU` → output `(batch, 64)`

### SkillActor (low-level, with GRU)
- Input: obs `(B, 11, 11, 13)`, team_skill_onehot `(B, N_Z_TEAM)`, indi_skill_onehot `(B, N_Z_INDI)`, rnn_state `(B, hidden_size)`
- CNN(obs) → concat with skills → Dense(hidden_size) → GRUCell(hidden_size) → Dense(64) → ReLU → Dense(9) → Categorical
- Output: pi (Categorical), new_rnn_state
- All agents share parameters

### SkillCritic (low-level, centralized, with GRU)
- Input: world_state `(B, 11, 11, 91)`, team_skill_onehot `(B, N_Z_TEAM)`, rnn_state `(B, hidden_size)`
- CNN(world_state) → concat with team_skill → Dense(hidden_size) → GRUCell(hidden_size) → Dense(64) → ReLU → Dense(1)
- Note: reference critic uses only team_skill, not indi_skill

### TeamDiscriminator (feedforward)
- Input: world_state `(B, 11, 11, 91)`
- CNN(world_state) → Dense(64) → ReLU → Dense(N_Z_TEAM) → logits
- Intrinsic reward = `log_softmax(logits)[Z]`

### IndividualDiscriminator (feedforward)
- Input: obs `(B, 11, 11, 13)`, team_skill_onehot `(B, N_Z_TEAM)`
- CNN(obs) → concat with team_skill → Dense(64) → ReLU → Dense(N_Z_INDI) → logits
- Intrinsic reward = `log_softmax(logits)[z^i]`

### SkillCoordinator (transformer, no RNN)
**Encoder:**
- Two separate CNNs: one for world_state (91ch), one for agent obs (13ch)
- CNN_ws(world_state) → Dense(n_embd) = global token
- CNN_obs(obs_i) → Dense(n_embd) = per-agent tokens (shared across agents via reshape+batch)
- Concatenate → LayerNorm → N_BLOCK × EncoderBlock(self-attention + FFN)
- Value head: Dense(n_embd) → GELU → LayerNorm → Dense(1) per token → `(B, n_agent+1, 1)`

**Decoder (autoregressive):**
- Input: shifted one-hot actions (START token, then previously sampled skills)
- Dense(n_embd) → LayerNorm → N_BLOCK × DecoderBlock(causal self-attention + cross-attention to encoder + FFN)
- Output: logits `(B, n_agent+1, max(N_Z_TEAM, N_Z_INDI))`
- Position 0 = team skill, positions 1..n = individual skills
- Generation: `jax.lax.scan` over n_agent+1 steps
- PPO evaluation: teacher-forcing parallel forward pass

## Training Loop Structure (hmasd_cnn_cleanup.py)

### NamedTuple Definitions (replacing HMASD's buffer classes)
```python
class LowTransition(NamedTuple):
    done: jnp.ndarray           # (NUM_ENVS*n_agents,)
    action: jnp.ndarray         # (NUM_ENVS*n_agents,)
    value: jnp.ndarray          # (NUM_ENVS*n_agents,)
    reward: jnp.ndarray         # (NUM_ENVS*n_agents,)  — combined reward
    log_prob: jnp.ndarray       # (NUM_ENVS*n_agents,)
    obs: jnp.ndarray            # (NUM_ENVS*n_agents, 11, 11, 13)
    rnn_state_actor: jnp.ndarray  # (NUM_ENVS*n_agents, hidden_size)
    rnn_state_critic: jnp.ndarray # (NUM_ENVS, hidden_size)  — centralized
    team_skill: jnp.ndarray     # (NUM_ENVS,)  — int index
    indi_skill: jnp.ndarray     # (NUM_ENVS*n_agents,)  — int index
    info: dict

class HighTransition(NamedTuple):
    done: jnp.ndarray           # (NUM_ENVS, n_agents+1)
    team_skill: jnp.ndarray     # (NUM_ENVS,)
    indi_skills: jnp.ndarray    # (NUM_ENVS, n_agents)
    value: jnp.ndarray          # (NUM_ENVS, n_agents+1)
    log_prob: jnp.ndarray       # (NUM_ENVS, n_agents+1)
    reward: jnp.ndarray         # (NUM_ENVS, n_agents+1)  — summed env reward
    world_state: jnp.ndarray    # (NUM_ENVS, 11, 11, 91)
    obs: jnp.ndarray            # (NUM_ENVS, n_agents, 11, 11, 13)
```

### Loop Pseudocode
```
make_train(config) → (init_runner_state, train_chunk, remainder_chunk, ...)

runner_state = (train_states, env_state, obsv, rnn_states_actor, rnn_states_critic, update_step, rng)

_update_step(runner_state, _):

  _skill_interval(carry, _):
    # carry = (env_state, obs, rnn_actor, rnn_critic, rng)
    rng, _rng = jax.random.split(rng)                 # ← RNG split for transformer
    1. Construct world_state from agent observations
    2. Skill Coordinator: team_skill Z, indi_skills z^{1:n}, values, log_probs (using _rng)
    3. GRU hidden states: carry through from previous interval (NOT reset at skill boundary)

    _env_step(inner_carry, _):
      # inner_carry = (env_state, obs, rnn_actor, rnn_critic, team_skill, indi_skills, rng)
      rng, _rng_act, _rng_step = jax.random.split(rng, 3)  # ← RNG splits for actor + env
      4. SkillActor: action from (obs, Z, z^i, rnn_state_actor) using _rng_act
      5. SkillCritic: value from (world_state, Z, rnn_state_critic)
      6. TeamDiscriminator: log q_D(Z|s) → team intrinsic reward
      7. IndividualDiscriminator: log q_d(z^i|o^i,Z) → indi intrinsic reward
      8. Step environment (using _rng_step, split per NUM_ENVS)
      9. Combined reward = λ_e * env_rew + λ_D * team_intri + λ_d * indi_intri
      10. Reset GRU states on episode done: rnn *= (1 - done)
      11. Return LowTransition (including rnn_states for PPO)
      return inner_carry, low_transition

    inner_scan(SKILL_INTERVAL) → low_transitions  # shape: (25, ...)
    12. High-level reward = sum(env_rewards over interval), avg across agents
    13. Return HighTransition
    return carry, (high_transition, low_transitions)

  outer_scan(SKILL_STEPS=40) → (h_transitions, l_transitions)
  # h_transitions shape: (40, ...), l_transitions shape: (40, 25, ...) → reshape to (1000, ...)

  # PPO Updates (rng split for each shuffle)
  rng, _rng = jax.random.split(rng)
  14. Low-level GAE → PPO for SkillActor + SkillCritic (with sequence chunking for RNN)
  rng, _rng = jax.random.split(rng)
  15. High-level GAE → PPO for SkillCoordinator (parallel mode)
  rng, _rng = jax.random.split(rng)
  16. Discriminator supervised cross-entropy updates (data from low_transitions)

  return runner_state, metrics
```

## Reward Structure
- **Low-level**: `r^i_t = λ_e * r_t + λ_D * log q_D(Z|s_{t+1}) + λ_d * log q_d(z^i|o^i_{t+1}, Z)`
- **High-level**: `r^h = Σ_{p=0}^{k-1} r_{t+p}` averaged across agents, broadcast to n_agent+1 positions

## Config (hmasd_cnn_cleanup.yaml)

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

## Milestones

### Milestone 1: Architecture Design Document (THIS SESSION)
- Detailed architecture mapping: every network with exact input/output shapes
- Detailed data flow: how observations, skills, rewards, RNN states flow through the nested scans
- Identify all JAX/Flax patterns needed (GRUCell, scan carry structure, transition types)
- Map reference PyTorch code to Flax equivalents line-by-line for critical components
- Output: the plan file you're reading, updated with any refinements

### Milestone 2: Network Definitions (hmasd_networks.py)
- CNN backbone
- SkillActor + SkillCritic (with GRUCell)
- TeamDiscriminator + IndividualDiscriminator
- Transformer Encoder + Decoder + autoregressive generation
- **Verify**: init with dummy inputs, forward pass, correct output shapes, no NaNs

### Milestone 3: Training Loop Skeleton + Basic Env Integration
- make_train() with init_runner_state
- Nested scan structure (_skill_interval outer, _env_step inner)
- GRU hidden state management in scan carry
- Intrinsic reward computation
- Combined reward and high-level reward aggregation
- Basic env stepping with cleanup environment
- **Verify**: single update step runs end-to-end, transitions have correct shapes

### Milestone 4: PPO + Discriminator Updates
- Low-level GAE + PPO update (with RNN sequence handling)
- High-level GAE + PPO update (parallel mode for transformer)
- Discriminator cross-entropy updates
- **Verify**: losses finite, gradients flow, params change after update

### Milestone 5: Logging, Eval Chunking, WandB Integration
- _reduce_metric_dict matching IPPO/MAPPO style
- evaluate() function (hierarchical: skills → actions with GRU states)
- Chunked training (10 evals + remainder), JIT compilation
- single_run() with WandB init
- Hydra config entry point
- **Verify**: full loop for small config, metrics in WandB, eval GIFs

### Milestone 6 (future): Harvest Open + Coins
- Adapt to harvest_open and coins environments
- Mainly config changes + env-specific metrics

## Potential Risks

1. **Memory pressure**: Nested scans store transitions including RNN states. 1000 × 32 × 7 × 64 (hidden) = ~14M floats per RNN state field. May need to reduce NUM_ENVS.
2. **RNN + PPO minibatching**: Need sequence-chunked minibatches to maintain RNN temporal consistency during PPO updates. This is the trickiest JAX pattern.
3. **Autoregressive decoding speed**: 8 sequential decoder calls per skill assignment (every 25 steps). Manageable.
4. **Reshape correctness**: Multiple agent-major vs env-major conversions plus RNN states. Use chex.assert_shape during development.
5. **Hyperparameter sensitivity**: Paper notes performance varies with skill counts and lambda values. May need tuning for cleanup.

## Critical Reference Files
- `algorithms/MAPPO/mappo_cnn_cleanup.py` — Primary template for training loop, world state, PPO, eval
- `algorithms/IPPO/ippo_cnn_cleanup.py` — CNN architecture, parameter sharing, metric reduction
- `NeurIPS2023_HMASD_code/hmasd/runner/shared/overcooked_runner.py` — HMASD episode loop, skill assignment flow
- `NeurIPS2023_HMASD_code/hmasd/algorithms/mat/algorithm/ma_transformer.py` — Transformer architecture
- `NeurIPS2023_HMASD_code/hmasd/algorithms/r_mappo/algorithm/r_actor_critic.py` — Actor/Critic with GRU + skill conditioning
- `NeurIPS2023_HMASD_code/hmasd/algorithms/discriminator/algorithm/discri_model.py` — Discriminator + intrinsic rewards
- `NeurIPS2023_HMASD_code/hmasd/config.py` — All HMASD hyperparameter defaults
- `NeurIPS2023_HMASD_code/hmasd/utils/l_shared_buffer.py` — Low-level buffer structure (shows RNN state storage)