"""
Network Definitions for our CTME algorithm


Networks are CNN-based to match SocialJax env style.

Components:
  - CNN: shared 3-layer CNN backbone (from IPPO)
  - SkillActor: low-level actor, CNN + 2 Dense layers, conditioned on observation + skills
  - SkillDiscriminator: skill predictor, CNN + 2 Dense layers
  - SkillCoordinator: high-level skill assignment (PPO-based)
"""

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp
import distrax

# ============================================================================
# CNN Backbone (from IPPO cleanup)
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
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        # flatten
        x = x.reshape((x.shape[0], -1)) 
        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        return x  # (batch, 64)
    

# ============================================================================
# SkillActor
# ============================================================================

class SkillActor(nn.Module):
    """
    Low-level agent skill executor.

    CNN(obs) -> concat(features, skill) -> Dense(64) -> activation -> 
    Dense(action_dim) -> Categorical

    * skill is represented as one-hot vector w/ shape (n_z)
    """
    action_dim: int             # TODO: should this be Sequence[int]?
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, skill):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        features = CNN(self.activation)(obs)                    # (batch, 64)
        x = jnp.concatenate([features, skill], axis=-1)         # (batch, 64 + n_z)
        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(x)                                                    # (batch, 64)
        x = activation(x)
        logits = nn.Dense(
            features=self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(x)                                                    # (batch, action_dim)
        pi = distrax.Categorical(logits=logits)
        return pi


# ============================================================================
# SkillCritic
# ============================================================================

class SkillCritic(nn.Module):
    """
    Low-level agent skill critic.
    
    CNN(obs) -> concat(features, skill) -> Dense(64) -> activation ->
    Dense(1)
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, skill):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        features = CNN(self.activation)(obs)                    # (batch, 64)
        x = jnp.concatenate([features, skill], axis=-1)         # (batch, 64 + n_z)
        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(x)                                                    # (batch, 64)
        x = activation(x)
        value = nn.Dense(
            1,
            # TODO: should this be 1.0 to match IPPO or keep 0.01 with MAPPO?
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)                                                    # (batch, 1)
        return jnp.squeeze(value, axis=-1)

# ============================================================================
# SkillDiscriminator
# ============================================================================

class SkillDiscriminator(nn.Module):
    """
    Takes in an agent's observation and predicts the skill.

    CNN(obs) -> Dense(64) -> activation -> Dense(n_z)
    """
    n_z: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        features = CNN(self.activation)(obs)        # (batch, 64)
        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(features)                                 # (batch, 64)
        x = activation(x)
        logits = nn.Dense(
            features=self.n_z, 
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(x)                                        # (batch, n_z)          
        return logits


# ============================================================================
# SkillCoordinator
# ============================================================================