"""
Loss function for policy optimization

Loss = Σ[w_E*(E_t/E_0) - w_G*(G_t/G_0)] + λ*(CO2_T/CO2_0)²

This encodes our objectives:
- Minimize emissions (w_E term)
- Maximize GDP (negative w_G term)
- Hit climate target (terminal penalty λ)

CRITICAL DESIGN DECISIONS:
1. Normalization: Divide by baseline (E_0, G_0) for scale invariance
2. Terminal penalty: Quadratic in final CO2 (heavily penalize missing target)
3. Running cost: Linear in emissions/GDP (simple, interpretable)

QUESTIONS:
- Should we normalize by city size? (YES - via E_0, G_0 from city config)
- Should terminal penalty be quadratic? (Yes - convex, smooth gradients)
- Should we add inequality constraints? (Use projection layer instead)
- Should we discount future? (Could add γ^t discount - advanced)

MULTI-CITY CONSIDERATIONS:
- Same loss structure for all cities (fair comparison)
- Different baselines (E_0, G_0) per city
- Same weights (w_E, w_G, λ) → what if we want city-specific?
"""

import jax.numpy as jnp

def compute_loss(trajectory, params, loss_weights):
    """
    Compute total loss over trajectory
    
    Loss = Σ_{t=0}^{T-1} [w_E*(E_t/E_0) - w_G*(G_t/G_0)] + λ*(CO2_T/CO2_0)²
    
    Args:
        trajectory: List of state dicts (length T+1)
        params: Parameter dict with E_0, G_0, CO2_0
        loss_weights: Dict with w_E, w_G, lambda
    
    Returns:
        float: Total loss (lower is better)
    
    COMPONENTS:
    1. Running cost: emissions penalty + GDP reward
    2. Terminal cost: final CO2 penalty
    
    QUESTIONS:
    - Should we weight early vs. late years differently? (Could use discount)
    - Should we add constraints (e.g., GDP never falls)? (Soft penalty)
    - How do we handle cities with different scales? (Normalization)
    
    NORMALIZATION EXPLANATION:
    E_t/E_0 = 1.0 means "same emissions as baseline"
    E_t/E_0 = 0.5 means "50% reduction" 
    G_t/G_0 = 2.0 means "GDP doubled"
    
    This makes loss comparable across cities!
    """
    w_E = loss_weights['w_E']
    w_G = loss_weights['w_G']
    lambda_term = loss_weights['lambda']
    
    # Baselines (from city config)
    E_0 = params['E_0']
    G_0 = params['G_0']
    CO2_0 = params['CO2_0']
    
    total_loss = 0.0
    
    # ══════════════════════════════════════════════════════════
    # RUNNING COST (over all timesteps except initial)
    # ══════════════════════════════════════════════════════════
    for state in trajectory[1:-1]:  # Exclude initial and final
        # Emissions penalty (normalized)
        emissions_penalty = w_E * (state['emissions'] / E_0)
        
        # GDP reward (negative = we want to maximize)
        gdp_reward = -w_G * (state['GDP'] / G_0)
        
        # Combine
        timestep_cost = emissions_penalty + gdp_reward
        total_loss += timestep_cost
    
    # ══════════════════════════════════════════════════════════
    # TERMINAL PENALTY (final CO2)
    # ══════════════════════════════════════════════════════════
    final_state = trajectory[-1]
    final_CO2 = final_state['CO2']
    
    # Quadratic penalty: want CO2_T ≈ CO2_0
    # (CO2_T/CO2_0 - 1)² penalizes deviations from baseline
    # OR just (CO2_T/CO2_0)² if we want absolute penalty
    terminal_penalty = lambda_term * (final_CO2 / CO2_0) ** 2
    
    total_loss += terminal_penalty
    
    return total_loss


def compute_loss_components(trajectory, params, loss_weights):
    """
    Break down loss into components for analysis
    
    Useful for understanding what drives the loss:
    - How much is emissions vs. GDP?
    - How much is running cost vs. terminal?
    - Which timesteps contribute most?
    
    Returns:
        dict with breakdown
    
    EXAMPLE OUTPUT:
        {
            'total_loss': 342.5,
            'emissions_cost': 245.2,
            'gdp_cost': -52.3,  # Negative because it's a reward
            'terminal_cost': 149.6,
            'avg_emissions_normalized': 0.85,  # 85% of baseline
            'avg_gdp_normalized': 1.32,  # 32% growth
            'final_co2_normalized': 1.08  # 8% above baseline
        }
    """
    w_E = loss_weights['w_E']
    w_G = loss_weights['w_G']
    lambda_term = loss_weights['lambda']
    
    E_0 = params['E_0']
    G_0 = params['G_0']
    CO2_0 = params['CO2_0']
    
    # Running costs
    emissions_cost = 0.0
    gdp_cost = 0.0
    
    emissions_list = []
    gdp_list = []
    
    for state in trajectory[1:-1]:
        emissions_penalty = w_E * (state['emissions'] / E_0)
        gdp_reward = -w_G * (state['GDP'] / G_0)
        
        emissions_cost += emissions_penalty
        gdp_cost += gdp_reward
        
        emissions_list.append(state['emissions'] / E_0)
        gdp_list.append(state['GDP'] / G_0)
    
    # Terminal cost
    final_CO2 = trajectory[-1]['CO2']
    terminal_cost = lambda_term * (final_CO2 / CO2_0) ** 2
    
    # Total
    total_loss = emissions_cost + gdp_cost + terminal_cost
    
    return {
        'total_loss': float(total_loss),
        'emissions_cost': float(emissions_cost),
        'gdp_cost': float(gdp_cost),
        'terminal_cost': float(terminal_cost),
        'avg_emissions_normalized': float(jnp.mean(jnp.array(emissions_list))),
        'avg_gdp_normalized': float(jnp.mean(jnp.array(gdp_list))),
        'final_co2_normalized': float(final_CO2 / CO2_0),
    }


def multi_objective_loss(trajectory, params, weights_list):
    """
    Compute loss for multiple objectives (Pareto optimization)
    
    Instead of single scalar weights, we can optimize for
    different trade-offs simultaneously.
    
    Args:
        trajectory: State sequence
        params: Parameters
        weights_list: List of weight dicts, e.g.
            [
                {'w_E': 5.0, 'w_G': 1.0, 'lambda': 20},  # Emissions focus
                {'w_E': 1.0, 'w_G': 5.0, 'lambda': 10},  # Growth focus
            ]
    
    Returns:
        array of losses, one per weight configuration
    
    USE CASE:
    Find Pareto frontier: what's the best GDP for each emissions level?
    
    QUESTIONS:
    - Should we use scalarization vs. true multi-objective? (Scalarization is simpler)
    - How do we pick weight combinations? (Grid search, adaptive)
    """
    losses = []
    for weights in weights_list:
        loss = compute_loss(trajectory, params, weights)
        losses.append(loss)
    
    return jnp.array(losses)


def feasibility_penalty(trajectory, params):
    """
    Add penalty for infeasible trajectories
    
    SOFT CONSTRAINTS:
    - GDP should never fall below 50% of baseline
    - CO2 should not exceed 2x baseline
    - Emissions should be non-negative
    
    These are SOFT (via penalty) not HARD (projection)
    
    QUESTIONS:
    - Do we need this? (Depends on whether dynamics can violate)
    - How large should penalties be? (Large enough to matter)
    - Should penalties be quadratic? (Yes for smoothness)
    """
    penalty = 0.0
    
    G_0 = params['G_0']
    CO2_0 = params['CO2_0']
    
    for state in trajectory:
        # GDP should not collapse
        if state['GDP'] < 0.5 * G_0:
            penalty += 1000 * (0.5 * G_0 - state['GDP'])**2
        
        # CO2 should not explode
        if state['CO2'] > 2.0 * CO2_0:
            penalty += 100 * (state['CO2'] - 2.0 * CO2_0)**2
        
        # Emissions should be non-negative (shouldn't happen)
        if state.get('emissions', 0) < 0:
            penalty += 1000 * state['emissions']**2
    
    return penalty


if __name__ == "__main__":
    """
    Test loss computation with dummy trajectory
    """
    # Create dummy trajectory
    trajectory = [
        {'CO2': 420, 'GDP': 100, 'emissions': 40, 't': 0},
        {'CO2': 425, 'GDP': 103, 'emissions': 38, 't': 1},
        {'CO2': 430, 'GDP': 106, 'emissions': 36, 't': 2},
        {'CO2': 435, 'GDP': 109, 'emissions': 34, 't': 3},
    ]
    
    params = {
        'E_0': 40.0,
        'G_0': 100.0,
        'CO2_0': 420.0
    }
    
    loss_weights = {
        'w_E': 1.0,
        'w_G': 1.0,
        'lambda': 10.0
    }
    
    # Compute loss
    loss = compute_loss(trajectory, params, loss_weights)
    breakdown = compute_loss_components(trajectory, params, loss_weights)
    
    print("\n" + "="*60)
    print("LOSS COMPUTATION TEST")
    print("="*60)
    print(f"\nTotal loss: {loss:.2f}")
    print(f"\nBreakdown:")
    for key, value in breakdown.items():
        print(f"  {key:30s}: {value:8.2f}")