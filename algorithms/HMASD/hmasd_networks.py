"""
HMASD Network Definitions for SocialJax (JAX/Flax)

Ported from NeurIPS2023_HMASD_code reference implementation.
All networks use CNN backbones matching IPPO/MAPPO cleanup style.

Components:
  - CNN: shared 3-layer CNN backbone (from IPPO)
  - SkillActor: low-level actor with GRU, conditioned on skills
  - SkillCritic: low-level centralized critic with GRU
  - TeamDiscriminator: feedforward, classifies team skill from world state
  - IndividualDiscriminator: feedforward, classifies individual skill from obs + team skill
  - SkillCoordinator: transformer encoder-decoder for autoregressive skill assignment
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import distrax
import math
from flax.linen.initializers import constant, orthogonal
from typing import Tuple


# ============================================================================
# CNN Backbone (identical to IPPO cleanup)
# ============================================================================

class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        return x  # (batch, 64)


# ============================================================================
# SkillActor (low-level, with GRU)
# ============================================================================

class SkillActor(nn.Module):
    """Low-level actor: obs + skills → action distribution.

    Architecture (skill_last_layer=True mode from reference):
        CNN(obs) → Dense(hidden_size) → GRUCell → concat(gru_out, team_skill, indi_skill)
        → Dense(64) → ReLU → Dense(action_dim) → Categorical

    All agents share parameters.
    """
    action_dim: int
    hidden_size: int = 64
    n_z_team: int = 3
    n_z_indi: int = 3
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, team_skill_onehot, indi_skill_onehot, rnn_state):
        """
        Args:
            obs: (B, H, W, C) agent observation
            team_skill_onehot: (B, n_z_team)
            indi_skill_onehot: (B, n_z_indi)
            rnn_state: (B, hidden_size) GRU hidden state
        Returns:
            pi: distrax.Categorical distribution
            new_rnn_state: (B, hidden_size)
        """
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # CNN backbone → (B, 64)
        features = CNN(self.activation)(obs)

        # Project to GRU input size
        gru_input = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(features)
        gru_input = activation(gru_input)

        # GRU step
        new_rnn_state, _ = nn.GRUCell(features=self.hidden_size)(rnn_state, gru_input)

        # LayerNorm on GRU output for action head (reference rnn.py:79)
        # Raw new_rnn_state (without LN) is carried forward as hidden state
        gru_out = nn.LayerNorm()(new_rnn_state)

        # Concat normalized GRU output with skill embeddings (skill_last_layer=True)
        actor_features = jnp.concatenate(
            [gru_out, team_skill_onehot, indi_skill_onehot], axis=-1
        )

        # Action head
        actor_features = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_features)
        actor_features = activation(actor_features)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_features)

        pi = distrax.Categorical(logits=logits)
        return pi, new_rnn_state


# ============================================================================
# SkillCritic (low-level, centralized, with GRU)
# ============================================================================

class SkillCritic(nn.Module):
    """Low-level centralized critic: world_state + team_skill → value.

    Architecture (skill_last_layer=True, critic uses team_skill only):
        CNN(world_state) → Dense(hidden_size) → GRUCell
        → concat(gru_out, team_skill) → Dense(64) → ReLU → Dense(1)
    """
    hidden_size: int = 64
    n_z_team: int = 3
    activation: str = "relu"

    @nn.compact
    def __call__(self, world_state, team_skill_onehot, rnn_state):
        """
        Args:
            world_state: (B, H, W, C*n_agents) concatenated observations
            team_skill_onehot: (B, n_z_team)
            rnn_state: (B, hidden_size) GRU hidden state
        Returns:
            value: (B,)
            new_rnn_state: (B, hidden_size)
        """
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # CNN backbone
        features = CNN(self.activation)(world_state)

        # Project to GRU input size
        gru_input = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(features)
        gru_input = activation(gru_input)

        # GRU step
        new_rnn_state, _ = nn.GRUCell(features=self.hidden_size)(rnn_state, gru_input)

        # LayerNorm on GRU output for value head (reference rnn.py:79)
        # Raw new_rnn_state (without LN) is carried forward as hidden state
        gru_out = nn.LayerNorm()(new_rnn_state)

        # Concat normalized GRU output with team skill (skill_last_layer=True, critic uses team_skill only)
        critic_features = jnp.concatenate([gru_out, team_skill_onehot], axis=-1)

        # Value head
        critic_features = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_features)
        critic_features = activation(critic_features)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic_features)

        return jnp.squeeze(value, axis=-1), new_rnn_state


# ============================================================================
# Discriminators (feedforward, no RNN)
# ============================================================================

class TeamDiscriminator(nn.Module):
    """Classifies team skill Z from world state.

    CNN(world_state) → Dense(64) → ReLU → Dense(n_z_team) → logits
    Intrinsic reward = log_softmax(logits)[Z]
    """
    n_z_team: int = 3
    activation: str = "relu"

    @nn.compact
    def __call__(self, world_state):
        """
        Args:
            world_state: (B, H, W, C*n_agents)
        Returns:
            logits: (B, n_z_team)
        """
        activation = nn.relu if self.activation == "relu" else nn.tanh

        features = CNN(self.activation)(world_state)
        x = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(features)
        x = activation(x)
        logits = nn.Dense(
            self.n_z_team,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return logits


class IndividualDiscriminator(nn.Module):
    """Classifies individual skill z^i from (obs, team_skill).

    CNN(obs) → concat(features, team_skill_onehot) → Dense(64) → ReLU → Dense(n_z_indi) → logits
    Intrinsic reward = log_softmax(logits)[z^i]
    """
    n_z_indi: int = 3
    n_z_team: int = 3
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, team_skill_onehot):
        """
        Args:
            obs: (B, H, W, C)
            team_skill_onehot: (B, n_z_team)
        Returns:
            logits: (B, n_z_indi)
        """
        activation = nn.relu if self.activation == "relu" else nn.tanh

        features = CNN(self.activation)(obs)
        x = jnp.concatenate([features, team_skill_onehot], axis=-1)
        x = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        logits = nn.Dense(
            self.n_z_indi,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return logits


# ============================================================================
# Transformer Components for SkillCoordinator
# ============================================================================

def _relu_gain():
    """Gain for orthogonal init matching nn.init.calculate_gain('relu')."""
    return np.sqrt(2)


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional causal mask.

    Matches reference ma_transformer.py SelfAttention.
    """
    n_embd: int
    n_head: int
    n_agent: int
    masked: bool = False

    @nn.compact
    def __call__(self, key, value, query):
        """
        Args:
            key: (B, L, n_embd)
            value: (B, L, n_embd)
            query: (B, L, n_embd)
        Returns:
            output: (B, L, n_embd)
        """
        B, L, D = query.shape
        d_head = D // self.n_head

        # Q, K, V projections (all with small init gain=0.01, matching reference init_())
        k = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(key)
        q = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(query)
        v = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(value)

        # Reshape to multi-head: (B, L, D) → (B, n_head, L, d_head)
        k = k.reshape(B, L, self.n_head, d_head).transpose(0, 2, 1, 3)
        q = q.reshape(B, L, self.n_head, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, d_head).transpose(0, 2, 1, 3)

        # Attention scores
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(d_head))

        # Optional causal mask
        if self.masked:
            mask = jnp.tril(jnp.ones((self.n_agent + 2, self.n_agent + 2)))
            mask = mask[None, None, :L, :L]  # (1, 1, L, L)
            att = jnp.where(mask == 0, -1e10, att)

        att = nn.softmax(att, axis=-1)

        # Weighted sum
        y = att @ v  # (B, n_head, L, d_head)
        y = y.transpose(0, 2, 1, 3).reshape(B, L, D)

        # Output projection
        y = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(y)
        return y


class EncodeBlock(nn.Module):
    """Transformer encoder block (post-LayerNorm with residual).

    x = LN1(x + self_attn(x, x, x))
    x = LN2(x + FFN(x))
    """
    n_embd: int
    n_head: int
    n_agent: int

    @nn.compact
    def __call__(self, x):
        # Self-attention (unmasked)
        attn_out = SelfAttention(
            n_embd=self.n_embd, n_head=self.n_head, n_agent=self.n_agent, masked=False
        )(x, x, x)
        x = nn.LayerNorm()(x + attn_out)

        # FFN
        mlp_out = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(_relu_gain()),
            bias_init=constant(0.0),
        )(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(mlp_out)
        x = nn.LayerNorm()(x + mlp_out)
        return x


class DecodeBlock(nn.Module):
    """Transformer decoder block with cross-attention.

    Reference pattern (unusual — output is encoder rep, not decoder state):
        x = LN1(x + causal_self_attn(x, x, x))
        x = LN2(rep_enc + cross_attn(key=x, value=x, query=rep_enc))
        x = LN3(x + FFN(x))
    """
    n_embd: int
    n_head: int
    n_agent: int

    @nn.compact
    def __call__(self, x, rep_enc):
        """
        Args:
            x: decoder tokens (B, L, n_embd)
            rep_enc: encoder representation (B, L, n_embd)
        Returns:
            x: updated representation (B, L, n_embd)
        """
        # Causal self-attention on decoder tokens
        attn1_out = SelfAttention(
            n_embd=self.n_embd, n_head=self.n_head, n_agent=self.n_agent, masked=True
        )(x, x, x)
        x = nn.LayerNorm()(x + attn1_out)

        # Cross-attention: query=rep_enc, key=x, value=x
        # NOTE: residual is on rep_enc, and output replaces x
        attn2_out = SelfAttention(
            n_embd=self.n_embd, n_head=self.n_head, n_agent=self.n_agent, masked=True
        )(key=x, value=x, query=rep_enc)
        x = nn.LayerNorm()(rep_enc + attn2_out)

        # FFN
        mlp_out = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(_relu_gain()),
            bias_init=constant(0.0),
        )(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(mlp_out)
        x = nn.LayerNorm()(x + mlp_out)
        return x


# ============================================================================
# SkillCoordinator (Transformer encoder-decoder)
# ============================================================================

class SkillCoordinator(nn.Module):
    """High-level transformer that assigns team skill Z and individual skills z^{1:n}.

    Encoder:
        CNN_ws(world_state) → global token
        CNN_obs(obs_i) → per-agent tokens (shared CNN, batched)
        Concatenate → LayerNorm → EncodeBlocks → value head

    Decoder (autoregressive):
        Shifted one-hot skills → team/indi action encoders → LayerNorm → DecodeBlocks → action head
        Generation via jax.lax.scan; evaluation via teacher-forcing parallel pass

    Output sequence: [team_skill, z^1, z^2, ..., z^n] (n_agent+1 positions)
    """
    n_agents: int
    n_z_team: int = 3
    n_z_indi: int = 3
    n_block: int = 1
    n_embd: int = 64
    n_head: int = 1
    activation: str = "relu"

    def setup(self):
        self.act_dim = max(self.n_z_team, self.n_z_indi)

    @nn.compact
    def __call__(self, world_state, all_obs, actions):
        """Forward pass exercising all sub-modules (use for init + evaluate).

        Args:
            world_state: (B, H, W, C_ws)
            all_obs: (B, n_agents, H, W, C)
            actions: (B, n_agents+1) int32 skill indices
        Returns:
            log_probs, values, entropy (same as evaluate)
        """
        return self.evaluate(world_state, all_obs, actions)

    @nn.compact
    def _encode(self, world_state, all_obs):
        """Encode world state and agent observations.

        Args:
            world_state: (B, H, W, C_ws) concatenated agent observations
            all_obs: (B, n_agents, H, W, C) per-agent observations
        Returns:
            v_loc: (B, n_agents+1, 1) value estimates per token
            obs_rep: (B, n_agents+1, n_embd) encoder representation
        """
        B = world_state.shape[0]

        # Encode world state → global token (B, 1, n_embd)
        ws_features = CNN(self.activation, name="cnn_ws")(world_state)  # (B, 64)
        ws_emb = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(_relu_gain()),
            bias_init=constant(0.0),
            name="ws_proj",
        )(ws_features)
        ws_emb = nn.gelu(ws_emb)  # (B, n_embd)
        ws_emb = ws_emb[:, None, :]  # (B, 1, n_embd)

        # Encode per-agent observations → agent tokens (B, n_agents, n_embd)
        obs_flat = all_obs.reshape(B * self.n_agents, *all_obs.shape[2:])  # (B*n, H, W, C)
        obs_features = CNN(self.activation, name="cnn_obs")(obs_flat)  # (B*n, 64)
        obs_emb = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(_relu_gain()),
            bias_init=constant(0.0),
            name="obs_proj",
        )(obs_features)
        obs_emb = nn.gelu(obs_emb)  # (B*n, n_embd)
        obs_emb = obs_emb.reshape(B, self.n_agents, self.n_embd)  # (B, n_agents, n_embd)

        # Concatenate: [global_token, agent_tokens] → (B, n_agents+1, n_embd)
        x = jnp.concatenate([ws_emb, obs_emb], axis=1)

        # LayerNorm + encoder blocks
        x = nn.LayerNorm(name="enc_ln")(x)
        for i in range(self.n_block):
            x = EncodeBlock(
                n_embd=self.n_embd, n_head=self.n_head, n_agent=self.n_agents,
                name=f"enc_block_{i}",
            )(x)

        # Value head per token
        v = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(_relu_gain()),
            bias_init=constant(0.0),
            name="v_dense1",
        )(x)
        v = nn.gelu(v)
        v = nn.LayerNorm(name="v_ln")(v)
        v_loc = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="v_dense2",
        )(v)  # (B, n_agents+1, 1)

        return v_loc, x  # x is obs_rep

    @nn.compact
    def _decode(self, shifted_action, obs_rep):
        """Decode shifted action sequence given encoder representation.

        Args:
            shifted_action: (B, n_agents+1, act_dim+1) shifted one-hot with START token
            obs_rep: (B, n_agents+1, n_embd) encoder output
        Returns:
            logits: (B, n_agents+1, act_dim) action logits
        """
        # Split into team (positions 0,1) and individual (positions 2..n) action tokens
        # Reference: team_action, indi_action = torch.split(action, [2, n_agent-1], dim=1)
        team_action = shifted_action[:, :2, :]  # (B, 2, act_dim+1)
        indi_action = shifted_action[:, 2:, :]  # (B, n_agents-1, act_dim+1)

        # Separate action encoders (no bias, matching reference)
        team_emb = nn.Dense(
            self.n_embd,
            use_bias=False,
            kernel_init=orthogonal(_relu_gain()),
            name="team_action_enc",
        )(team_action)
        team_emb = nn.gelu(team_emb)  # (B, 2, n_embd)

        indi_emb = nn.Dense(
            self.n_embd,
            use_bias=False,
            kernel_init=orthogonal(_relu_gain()),
            name="indi_action_enc",
        )(indi_action)
        indi_emb = nn.gelu(indi_emb)  # (B, n_agents-1, n_embd)

        # Concatenate action embeddings
        x = jnp.concatenate([team_emb, indi_emb], axis=1)  # (B, n_agents+1, n_embd)

        # LayerNorm + decoder blocks
        x = nn.LayerNorm(name="dec_ln")(x)
        for i in range(self.n_block):
            x = DecodeBlock(
                n_embd=self.n_embd, n_head=self.n_head, n_agent=self.n_agents,
                name=f"dec_block_{i}",
            )(x, obs_rep)

        # Action head
        logits = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(_relu_gain()),
            bias_init=constant(0.0),
            name="act_dense1",
        )(x)
        logits = nn.gelu(logits)
        logits = nn.LayerNorm(name="act_ln")(logits)
        logits = nn.Dense(
            self.act_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="act_dense2",
        )(logits)  # (B, n_agents+1, act_dim)

        return logits

    def _build_available_actions_mask(self):
        """Build mask for when n_z_team != n_z_indi.

        Returns:
            mask: (n_agents+1, act_dim) boolean, True=available
        """
        mask = jnp.ones((self.n_agents + 1, self.act_dim), dtype=jnp.bool_)
        if self.n_z_team < self.act_dim:
            # Position 0 (team skill): mask out indices >= n_z_team
            mask = mask.at[0, self.n_z_team:].set(False)
        if self.n_z_indi < self.act_dim:
            # Positions 1..n (individual skills): mask out indices >= n_z_indi
            mask = mask.at[1:, self.n_z_indi:].set(False)
        return mask

    def get_actions(self, world_state, all_obs, rng):
        """Autoregressive skill generation (used during rollout).

        Args:
            world_state: (B, H, W, C_ws)
            all_obs: (B, n_agents, H, W, C)
            rng: JAX PRNG key
        Returns:
            actions: (B, n_agents+1) int32 skill indices
            log_probs: (B, n_agents+1) log probabilities
            values: (B, n_agents+1) value estimates
        """
        B = world_state.shape[0]
        n_seq = self.n_agents + 1

        v_loc, obs_rep = self._encode(world_state, all_obs)
        values = jnp.squeeze(v_loc, axis=-1)  # (B, n_agents+1)

        avail_mask = self._build_available_actions_mask()  # (n_agents+1, act_dim)

        # Initialize shifted action with START token
        shifted_action = jnp.zeros((B, n_seq, self.act_dim + 1))
        shifted_action = shifted_action.at[:, 0, 0].set(1.0)

        def _step(carry, i):
            shifted_action, rng = carry
            rng, _rng = jax.random.split(rng)

            # Full decoder forward (all positions), take position i
            logits = self._decode(shifted_action, obs_rep)  # (B, n_seq, act_dim)
            logit_i = logits[:, i, :]  # (B, act_dim)

            # Apply available actions mask
            mask_i = avail_mask[i]  # (act_dim,)
            logit_i = jnp.where(mask_i, logit_i, -1e10)

            # Sample
            dist = distrax.Categorical(logits=logit_i)
            action = dist.sample(seed=_rng)  # (B,)
            log_prob = dist.log_prob(action)  # (B,)

            # Update shifted action for next position
            one_hot = jax.nn.one_hot(action, self.act_dim)  # (B, act_dim)
            # shifted_action[:, i+1, 1:] = one_hot (only if i+1 < n_seq)
            next_pos = jnp.zeros((B, n_seq, self.act_dim + 1))
            next_pos = next_pos.at[:, i + 1, 1:].set(one_hot)
            # Use where to only apply when i+1 < n_seq
            do_update = (i + 1 < n_seq)
            shifted_action = jnp.where(do_update, shifted_action + next_pos, shifted_action)

            return (shifted_action, rng), (action, log_prob)

        (shifted_action, _), (actions, log_probs) = jax.lax.scan(
            _step,
            (shifted_action, rng),
            jnp.arange(n_seq),
        )
        # actions: (n_seq, B), log_probs: (n_seq, B) → transpose to (B, n_seq)
        actions = actions.transpose(1, 0)  # (B, n_agents+1)
        log_probs = log_probs.transpose(1, 0)  # (B, n_agents+1)

        return actions.astype(jnp.int32), log_probs, values

    def evaluate(self, world_state, all_obs, actions):
        """Teacher-forcing parallel evaluation (used during PPO update).

        Args:
            world_state: (B, H, W, C_ws)
            all_obs: (B, n_agents, H, W, C)
            actions: (B, n_agents+1) int32 skill indices from trajectory
        Returns:
            log_probs: (B, n_agents+1)
            values: (B, n_agents+1)
            entropy: (B, n_agents+1)
        """
        B = world_state.shape[0]
        n_seq = self.n_agents + 1

        v_loc, obs_rep = self._encode(world_state, all_obs)
        values = jnp.squeeze(v_loc, axis=-1)  # (B, n_agents+1)

        avail_mask = self._build_available_actions_mask()  # (n_agents+1, act_dim)

        # Build shifted action (teacher forcing)
        one_hot_action = jax.nn.one_hot(actions, self.act_dim)  # (B, n_seq, act_dim)
        shifted_action = jnp.zeros((B, n_seq, self.act_dim + 1))
        shifted_action = shifted_action.at[:, 0, 0].set(1.0)  # START token
        shifted_action = shifted_action.at[:, 1:, 1:].set(one_hot_action[:, :-1, :])

        # Single parallel forward pass
        logits = self._decode(shifted_action, obs_rep)  # (B, n_seq, act_dim)

        # Apply available actions mask
        avail_mask_broadcast = avail_mask[None, :, :]  # (1, n_seq, act_dim)
        logits = jnp.where(avail_mask_broadcast, logits, -1e10)

        # Compute log_prob and entropy at all positions
        dist = distrax.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)  # (B, n_agents+1)
        entropy = dist.entropy()  # (B, n_agents+1)

        return log_probs, values, entropy

    def get_values(self, world_state, all_obs):
        """Value-only inference (for bootstrapping).

        Args:
            world_state: (B, H, W, C_ws)
            all_obs: (B, n_agents, H, W, C)
        Returns:
            values: (B, n_agents+1)
        """
        v_loc, _ = self._encode(world_state, all_obs)
        return jnp.squeeze(v_loc, axis=-1)


# ============================================================================
# Intrinsic reward helpers
# ============================================================================

def compute_team_intrinsic_reward(logits, team_skill_idx):
    """Compute team discriminator intrinsic reward.

    Reference default (intri_rew_exp=1) applies exp(), giving softmax probability.
    Paper Eq.4 uses log q_D(Z|s), but the reference code that produced results uses exp().

    Args:
        logits: (B, n_z_team) from TeamDiscriminator
        team_skill_idx: (B,) int indices
    Returns:
        reward: (B,) softmax(logits)[Z] i.e. p(Z|s), matching reference default
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, n_z_team)
    log_reward = log_probs[jnp.arange(log_probs.shape[0]), team_skill_idx]  # (B,)
    return jnp.exp(log_reward)


def compute_indi_intrinsic_reward(logits, indi_skill_idx):
    """Compute individual discriminator intrinsic reward.

    Reference default (intri_rew_exp=1) applies exp(), giving softmax probability.

    Args:
        logits: (B, n_z_indi) from IndividualDiscriminator
        indi_skill_idx: (B,) int indices
    Returns:
        reward: (B,) softmax(logits)[z^i] i.e. p(z^i|o^i,Z), matching reference default
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_reward = log_probs[jnp.arange(log_probs.shape[0]), indi_skill_idx]
    return jnp.exp(log_reward)
