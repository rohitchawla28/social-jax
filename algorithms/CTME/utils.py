import os
import pickle

import jax
import jax.numpy as jnp

# ============================================================================
# Batchify / Unbatchify helpers (from SocialJax)
# ============================================================================

def batchify(x: dict, agent_list, num_actors):
    """Stack dict values by str key, reshape to (num_actors, -1).
    Used for done dict: {str(agent): (NUM_ENVS,)} → (NUM_ACTORS, 1) → squeeze.
    """
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_numpy(x, agent_list, num_actors):
    """Stack array slices by integer index, reshape to (num_actors, -1).
    Used for reward: (NUM_ENVS, n_agents) array → (NUM_ACTORS, 1) → squeeze.
    """
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    """Reshape (NUM_ACTORS,) → (num_agents, num_envs, -1), return as dict.
    """
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


# ============================================================================
# Save / Load Parameters helpers (from SocialJax)
# ============================================================================

def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)

# make_world_state
# sample_skills
# one_hot_skills
# compute_returns
# logging helpers
# checkpoint helpers