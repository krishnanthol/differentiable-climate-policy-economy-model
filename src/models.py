"""
Policy network architectures

This file defines how we parameterize policies:
- Neural network: state → action
- Time-varying policies: t → action
- Constant policies: single action for all time

CRITICAL DESIGN CHOICE:
We use a simple feedforward network that maps state → policy.
This allows policies to be ADAPTIVE (react to current state).

QUESTIONS:
- Should policy depend on state or just time? (State-dependent is more general)
- What network architecture? (MLP is simplest, works well)
- Should we use different networks for different cities? (Same arch, different weights)
- How do we ensure policies stay in [0,1]? (Sigmoid output OR projection layer)

DESIGN DECISIONS:
- Network is small (2 layers, 64 hidden) for 3-day training
- Sigmoid output ensures [0,1] without projection
- JAX-native implementation for autodiff
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple

def init_policy_params(key, state_dim=2, action_dim=3, hidden_dim=64):
    """
    Initialize neural network policy parameters
    
    Architecture:
        Input (state_dim=2): [CO2_normalized, GDP_normalized]
        Hidden layer 1: 64 units, tanh activation
        Hidden layer 2: 64 units, tanh activation
        Output (action_dim=3): [τ, s, c], sigmoid activation
    
    Args:
        key: JAX random key
        state_dim: Input dimension (2 for [CO2, GDP])
        action_dim: Output dimension (3 for [τ, s, c])
        hidden_dim: Hidden layer size
    
    Returns:
        dict: Network parameters {W1, b1, W2, b2, W3, b3}
    
    QUESTIONS:
    - Why tanh activation? (Centers activations, helps gradients)
    - Why sigmoid output? (Forces [0,1] range naturally)
    - Should we use batch norm? (Not needed for small network)
    - How to initialize weights? (Xavier/He - using small normal here)
    
    INITIALIZATION STRATEGY:
    - Small random weights (0.1 std) → prevent saturation
    - Small biases (0.01 std) → slight positive bias ok
    - Uniform initialization across cities → fair comparison
    """
    keys = jax.random.split(key, 6)
    
    # Layer 1: state → hidden
    W1 = jax.random.normal(keys[0], (state_dim, hidden_dim)) * 0.1
    b1 = jax.random.normal(keys[1], (hidden_dim,)) * 0.01
    
    # Layer 2: hidden → hidden
    W2 = jax.random.normal(keys[2], (hidden_dim, hidden_dim)) * 0.1
    b2 = jax.random.normal(keys[3], (hidden_dim,)) * 0.01
    
    # Layer 3: hidden → action
    W3 = jax.random.normal(keys[4], (hidden_dim, action_dim)) * 0.1
    b3 = jax.random.normal(keys[5], (action_dim,)) * 0.01
    
    params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3,
    }
    
    return params


def normalize_state(state, params):
    """
    Normalize state to [0, 1] range for network input
    
    Neural networks work better with normalized inputs.
    
    NORMALIZATION SCHEME:
    - CO2: (CO2_t - CO2_0) / (2 * CO2_0)  → roughly [0, 1] for reasonable futures
    - GDP: (GDP_t - GDP_0) / (2 * GDP_0)  → roughly [0, 1] for growth scenarios
    
    QUESTIONS:
    - Should we use running statistics? (Not needed - we know rough ranges)
    - What if values exceed [0, 1]? (OK - network can handle it)
    - Should normalization be city-specific? (Yes - using city's baseline)
    """
    CO2_normalized = (state['CO2'] - params['CO2_0']) / (2 * params['CO2_0'])
    GDP_normalized = (state['GDP'] - params['G_0']) / (2 * params['G_0'])
    
    # Clip to reasonable range to avoid extreme values
    CO2_normalized = jnp.clip(CO2_normalized, -1, 2)
    GDP_normalized = jnp.clip(GDP_normalized, -1, 2)
    
    return jnp.array([CO2_normalized, GDP_normalized])


def policy_network(params, state, state_params):
    """
    Feedforward neural network: state → action
    
    Args:
        params: Network parameters (weights & biases)
        state: Current state dict {CO2, GDP, ...}
        state_params: Parameters for normalization (CO2_0, G_0)
    
    Returns:
        action: [τ, s, c] in [0, 1]³
    
    FORWARD PASS:
        x = normalize_state(state)
        h1 = tanh(x @ W1 + b1)
        h2 = tanh(h1 @ W2 + b2)
        action = sigmoid(h2 @ W3 + b3)
    
    QUESTIONS:
    - Should we add skip connections? (Not needed for shallow network)
    - Should we add dropout? (No - we're not training on data, no overfitting)
    - What if we want deterministic policies? (This IS deterministic)
    """
    # Normalize input
    x = normalize_state(state, state_params)
    
    # Layer 1
    h1 = jnp.tanh(x @ params['W1'] + params['b1'])
    
    # Layer 2
    h2 = jnp.tanh(h1 @ params['W2'] + params['b2'])
    
    # Output layer (sigmoid → [0, 1])
    action_logits = h2 @ params['W3'] + params['b3']
    action = jnp.sigmoid(action_logits)
    
    return action


def get_trajectory_policies(policy_params, initial_state, T, state_params):
    """
    Get policy sequence for entire trajectory
    
    This creates a T-length sequence of policies by:
    1. Starting from initial_state
    2. Query policy network for action
    3. Use simple state projection for next query
    
    NOTE: This is NOT a full forward simulation!
    We're just creating a policy sequence to feed into the dynamics.
    
    APPROXIMATION:
    We use a simple state evolution for policy planning:
    - CO2 ≈ CO2_0 + 0.5*t (rough linear increase)
    - GDP ≈ GDP_0 * (1.02)^t (rough 2% growth)
    
    This is OK because:
    1. Policies will be optimized anyway
    2. Full dynamics happens in simulate_trajectory()
    3. This is just for initialization
    
    QUESTIONS:
    - Should we use full dynamics here? (No - too slow, unnecessary)
    - Should policy depend on actual state? (Yes - but we approximate)
    - What if state evolution is way off? (Optimizer will fix it)
    
    ALTERNATIVE APPROACHES:
    1. Time-varying policy: policy[t] = f(t) [simpler]
    2. State-dependent: policy = f(state) [what we do]
    3. Recurrent: policy[t] = f(state, policy[t-1]) [more complex]
    """
    policies = []
    state = initial_state
    
    for t in range(T):
        # Get action from policy network
        action = policy_network(policy_params, state, state_params)
        policies.append(action)
        
        # Simple state projection for next policy query
        # (NOT the actual dynamics - just for planning)
        state = {
            'CO2': state['CO2'] + 0.5,  # Rough CO2 increase
            'GDP': state['GDP'] * 1.02,  # Rough 2% growth
            't': t + 1
        }
    
    return jnp.array(policies)


def constant_policy(value):
    """
    Simple baseline: constant policy for all time
    
    Useful for:
    - Baselines (e.g., τ=0.5 always)
    - Testing
    - Comparison to learned policies
    
    Args:
        value: [τ, s, c] constant
    
    Returns:
        function that returns same policy regardless of state
    
    EXAMPLE:
        no_policy = constant_policy([0, 0, 0])
        aggressive_tax = constant_policy([0.8, 0.3, 0.5])
    """
    def policy_fn(params, state, state_params):
        return jnp.array(value)
    
    return policy_fn


def time_varying_policy(schedule):
    """
    Policy that changes over time (not state-dependent)
    
    Useful for:
    - Ramp-up schedules (e.g., τ increases linearly)
    - Step functions (e.g., sudden carbon tax at t=10)
    
    Args:
        schedule: function t → [τ, s, c]
    
    EXAMPLE:
        def ramp_up(t):
            tau = min(t / 50, 0.8)  # Ramp to 0.8 over 50 years
            return [tau, 0.3, 0.2]
        
        policy = time_varying_policy(ramp_up)
    """
    def policy_fn(params, state, state_params):
        t = state['t']
        return jnp.array(schedule(t))
    
    return policy_fn


# ═══════════════════════════════════════════════════════════════
# POLICY VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════

def count_parameters(params):
    """
    Count total trainable parameters
    
    QUESTIONS:
    - How many parameters is too many? (>10k might overfit)
    - Does network size matter for different cities? (No - same arch)
    """
    total = 0
    for key, value in params.items():
        total += value.size
    return total


def policy_summary(policy_params):
    """
    Print summary of policy network
    
    Useful for debugging and reporting
    """
    n_params = count_parameters(policy_params)
    
    print("\n" + "="*60)
    print("POLICY NETWORK SUMMARY")
    print("="*60)
    print(f"Architecture: [2] → [64] → [64] → [3]")
    print(f"Total parameters: {n_params:,}")
    print(f"Activation: tanh (hidden), sigmoid (output)")
    print(f"Output range: [0, 1]³")
    print("="*60)


if __name__ == "__main__":
    """
    Test policy network initialization and forward pass
    """
    # Initialize
    key = jax.random.PRNGKey(42)
    params = init_policy_params(key)
    
    # Print summary
    policy_summary(params)
    
    # Test forward pass
    test_state = {'CO2': 420.0, 'GDP': 100.0, 't': 0}
    test_state_params = {'CO2_0': 420.0, 'G_0': 100.0}
    
    action = policy_network(params, test_state, test_state_params)
    
    print(f"\nTest forward pass:")
    print(f"  Input state: CO2={test_state['CO2']}, GDP={test_state['GDP']}")
    print(f"  Output action: τ={action[0]:.3f}, s={action[1]:.3f}, c={action[2]:.3f}")
    print(f"  ✓ All values in [0, 1]: {jnp.all((action >= 0) & (action <= 1))}")