## Milestone 2 Implementation Notes

These notes pre-digest the reference code so a new session can implement `hmasd_networks.py` without extensive re-exploration.

### Flax/JAX Patterns to Reuse from IPPO

**Orthogonal initialization scales (from `ippo_cnn_cleanup.py`):**
```python
from flax.linen.initializers import constant, orthogonal
import numpy as np

kernel_init=orthogonal(np.sqrt(2))   # Conv layers, hidden Dense layers
kernel_init=orthogonal(0.01)         # Final action logit layer (small scale)
kernel_init=orthogonal(1.0)          # Value output layer
bias_init=constant(0.0)              # All biases
```

**CNN backbone (copy from IPPO, `ippo_cnn_cleanup.py:29-66`):**
```python
class CNN(nn.Module):
    activation: str = "relu"
    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(32, (5,5), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(32, (3,3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(32, (3,3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        return x  # (batch, 64)
```

**Action distribution:** `distrax.Categorical(logits=logits)` — sample with `.sample(seed=rng)`, evaluate with `.log_prob(action)`, `.entropy()`.

**Parameter sharing reshape (from IPPO):**
```python
# Observations: (NUM_ENVS, num_agents, H, W, C) → (num_agents*NUM_ENVS, H, W, C)
obs_batch = jnp.transpose(last_obs, (1,0,2,3,4)).reshape(-1, *obs_shape)
# This puts agents as the primary axis (agent-major), then flattens

# Actions back: (num_agents*NUM_ENVS,) → dict of agent→(NUM_ENVS,)
env_act = unbatchify(action, env.agents, NUM_ENVS, num_agents)
```

### GRU in Flax

The reference uses `nn.GRU` (PyTorch) with `num_layers=1` (default), orthogonal weight init, zero bias init, followed by LayerNorm. In Flax:

```python
# Flax equivalent: nn.GRUCell for single-step operation inside jax.lax.scan
gru_cell = nn.GRUCell(features=hidden_size)
# Usage: new_carry, output = gru_cell(carry, input)
# carry = hidden state (batch, hidden_size), input = (batch, input_size)
# output = new_carry (they're the same for GRU)
```

**Hidden state reset on episode done:**
```python
# In the inner scan, after getting done signal:
rnn_state = rnn_state * (1 - done_expanded)  # Zero out on done
```

**For PPO with RNN sequences:** Store the initial RNN state at the start of each skill interval (or start of sequence chunk) in the transition. During PPO update, re-run the GRU from the stored initial state through the sequence to get consistent hidden states.

### Skill Conditioning Pattern

From reference `r_actor_critic.py` with `skill_last_layer=True` (our chosen mode):

```
# Actor: CNN(obs) → features(64) → GRU → concat(features, team_skill_onehot, indi_skill_onehot) → action_head
# Critic: CNN(world_state) → features(64) → GRU → concat(features, team_skill_onehot) → value_head
```

Skills are one-hot encoded and concatenated AFTER the GRU, right before the output head. This means:
- GRU input size = hidden_size (from a Dense layer that maps CNN output to hidden_size)
- Action head input size = hidden_size + N_Z_TEAM + N_Z_INDI
- Value head input size = hidden_size + N_Z_TEAM

### Transformer Architecture Details

This is the most complex component. The reference is in `ma_transformer.py`.

**Overall structure:**
```
MultiAgentTransformer
├── Encoder
│   ├── state_encoder: LayerNorm → Dense(n_embd, activate) → GELU  [for world_state]
│   ├── obs_encoder: LayerNorm → Dense(n_embd, activate) → GELU    [for per-agent obs]
│   ├── N_BLOCK × EncodeBlock (self-attention + FFN)
│   └── value_head: Dense(n_embd, activate) → GELU → LayerNorm → Dense(1)
└── Decoder
    ├── team_action_encoder: Dense(act_dim+1 → n_embd, no bias) → GELU  [position 0]
    ├── indi_action_encoder: Dense(act_dim+1 → n_embd, no bias) → GELU  [positions 1..n]
    ├── N_BLOCK × DecodeBlock (causal self-attention + cross-attention + FFN)
    └── action_head: Dense(n_embd) → GELU → LayerNorm → Dense(act_dim)
```

**EncodeBlock (pre-LayerNorm with residuals):**
```
x → LN1 → SelfAttention(q=x, k=x, v=x, masked=False) → + residual(x)
  → LN2 → FFN(Dense→GELU→Dense) → + residual
```

**DecodeBlock (pre-LayerNorm, asymmetric cross-attention):**
```
x → LN1 → CausalSelfAttention(q=x, k=x, v=x, masked=True) → + residual(x)
  → LN2(x) → CrossAttention(query=rep_enc, key=x, value=x) → + residual(rep_enc)  [NOTE: residual on rep_enc!]
  → LN3(rep_enc) → FFN(Dense→GELU→Dense) → + residual(rep_enc)
Output: rep_enc (NOT x)
```

**Critical detail**: Cross-attention uses encoder output (`rep_enc`) as the query, and decoder state (`x`) as key/value. The residual connection and output are on `rep_enc`, not `x`. This is unusual — most transformers do it the other way. Must preserve this.

**Causal masking:**
```python
# Lower triangular mask: position i can only attend to positions 0..i
mask = jnp.tril(jnp.ones((n_agent+2, n_agent+2)))[None, None, :, :]
# Applied: att = jnp.where(mask[:, :, :L, :L] == 0, -1e10, att)
```

**Self-attention implementation:**
```python
# Q, K, V projections: Dense(n_embd → n_embd) each
# Split into heads: reshape (B, L, n_embd) → (B, n_head, L, n_embd//n_head)
# Attention: softmax(Q @ K^T / sqrt(d_k)) @ V
# Optional causal mask before softmax
# Recombine heads: reshape back to (B, L, n_embd)
# Output projection: Dense(n_embd → n_embd)
```

**Initialization in transformer:**
- Weight init: `orthogonal_` with gain (default 0.01, or ReLU gain ~1.43 when `activate=True`)
- Bias init: constant 0
- Action encoder Dense layers: `use_bias=False`

**Encoder input construction:**
```python
# state_encoder processes world_state → (B, n_embd) → this is the "team" token
# obs_encoder processes each agent's obs → (B, n_agent, n_embd) → agent tokens
# In reference: state rep is averaged across agent dim, then expanded
# Concatenate: [state_token, agent_tokens] → (B, n_agent+1, n_embd)
# Pass through LayerNorm, then EncodeBlocks
```

### Autoregressive Decoding (from `transformer_act.py`)

**Generation mode (used during rollout):**
```
1. Initialize shifted_action = zeros(B, n_agent+1, act_dim+1)
2. Set START token: shifted_action[:, 0, 0] = 1  (one-hot indicator)
3. For i in 0..n_agent:
   a. logits = decoder(shifted_action, encoder_output)[:, i, :]  → (B, act_dim)
   b. Apply available_actions mask (team at pos 0 may have different valid actions than agents)
   c. Sample: action_i ~ Categorical(logits)
   d. Store action_i and log_prob_i
   e. If i+1 < n_agent+1: shifted_action[:, i+1, 1:] = one_hot(action_i)
4. Return: actions (B, n_agent+1, 1), log_probs (B, n_agent+1, 1)
```

In JAX, step 3 becomes `jax.lax.scan` over `n_agent+1` steps. The carry holds `(shifted_action, rng)`.

**Teacher-forcing mode (used during PPO evaluation):**
```
1. Given actions from trajectory: (B, n_agent+1, 1)
2. One-hot encode: (B, n_agent+1, act_dim)
3. Build shifted_action: START token at pos 0, shifted one-hot at pos 1..n
   shifted_action[:, 0, 0] = 1
   shifted_action[:, 1:, 1:] = one_hot_actions[:, :-1, :]
4. Single forward pass through decoder: logits = decoder(shifted_action, encoder_output)
5. Evaluate all positions at once: log_probs, entropy from Categorical(logits)
```

This is a single parallel forward pass — no scan needed.

**Available actions mask (when N_Z_TEAM ≠ N_Z_INDI):**
```python
# act_dim = max(N_Z_TEAM, N_Z_INDI)
# If N_Z_TEAM < N_Z_INDI: mask positions N_Z_TEAM..act_dim-1 for team (pos 0)
# If N_Z_TEAM > N_Z_INDI: mask positions N_Z_INDI..act_dim-1 for agents (pos 1..n)
# With default N_Z_TEAM=3, N_Z_INDI=3: no masking needed
```

### Module Dependency Order for Implementation

Implement in this order (each builds on the previous):
1. `CNN` — standalone, copy from IPPO
2. `SkillActor` — uses CNN, adds GRU + skill concat + action head
3. `SkillCritic` — uses CNN, adds GRU + skill concat + value head
4. `TeamDiscriminator` — uses CNN, simple feedforward
5. `IndividualDiscriminator` — uses CNN, feedforward with skill concat
6. `SelfAttention` — standalone attention module with optional causal mask
7. `EncodeBlock` / `DecodeBlock` — uses SelfAttention
8. `SkillCoordinator` — uses CNN + EncodeBlock + DecodeBlock + autoregressive scan

### Verification Script for Milestone 2

After implementing all modules, verify with:
```python
import jax
import jax.numpy as jnp
from hmasd_networks import *

rng = jax.random.PRNGKey(0)
B, n_agents = 4, 7  # small batch for testing

# Test CNN
cnn = CNN()
dummy_obs = jnp.zeros((B, 11, 11, 13))
params = cnn.init(rng, dummy_obs)
out = cnn.apply(params, dummy_obs)
assert out.shape == (B, 64), f"CNN: {out.shape}"

# Test SkillActor
actor = SkillActor(action_dim=9, hidden_size=64, n_z_team=3, n_z_indi=3)
dummy_team = jnp.zeros((B, 3))
dummy_indi = jnp.zeros((B, 3))
dummy_rnn = jnp.zeros((B, 64))
params = actor.init(rng, dummy_obs, dummy_team, dummy_indi, dummy_rnn)
pi, new_rnn = actor.apply(params, dummy_obs, dummy_team, dummy_indi, dummy_rnn)
assert new_rnn.shape == (B, 64)

# Test SkillCritic
critic = SkillCritic(hidden_size=64, n_z_team=3)
dummy_ws = jnp.zeros((B, 11, 11, 91))
params = critic.init(rng, dummy_ws, dummy_team, dummy_rnn)
val, new_rnn = critic.apply(params, dummy_ws, dummy_team, dummy_rnn)
assert val.shape == (B,)

# Test discriminators
team_disc = TeamDiscriminator(n_z_team=3)
params = team_disc.init(rng, dummy_ws)
logits = team_disc.apply(params, dummy_ws)
assert logits.shape == (B, 3)

indi_disc = IndividualDiscriminator(n_z_indi=3, n_z_team=3)
params = indi_disc.init(rng, dummy_obs, dummy_team)
logits = indi_disc.apply(params, dummy_obs, dummy_team)
assert logits.shape == (B, 3)

# Test SkillCoordinator
coord = SkillCoordinator(n_agents=7, n_z_team=3, n_z_indi=3, n_block=1, n_embd=64, n_head=1)
dummy_all_obs = jnp.zeros((B, 7, 11, 11, 13))
params = coord.init(rng, dummy_ws, dummy_all_obs)
# Test autoregressive generation
actions, log_probs, values = coord.apply(params, dummy_ws, dummy_all_obs, rng=rng, method=coord.get_actions)
assert actions.shape == (B, 8, 1)  # n_agent+1
assert values.shape == (B, 8, 1)
# Test teacher-forcing evaluation
dummy_actions = jnp.zeros((B, 8, 1), dtype=jnp.int32)
log_probs, values, entropy = coord.apply(params, dummy_ws, dummy_all_obs, dummy_actions, method=coord.evaluate)

print("All shape checks passed!")
```
