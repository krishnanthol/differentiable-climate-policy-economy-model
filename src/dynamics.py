"""
Core climate-economic-traffic dynamics

This file implements the forward simulation of the coupled system:
  Policies → Traffic → Emissions → CO2 → Climate Damage → GDP

KEY EQUATIONS:
1. Emissions: E_t = A * exp(-η_τ*τ - η_c*c) * (1-κ*s) * (G/G_0)^γ
2. CO2: CO2_{t+1} = CO2_t + α*E_t - δ*CO2_t  
3. GDP: G_{t+1} = G_t*(1+g_0)*(1+β_s*s)*(1-β_τ*τ)*(1-β_c*c)*(1-θ*(CO2/CO2_0)^φ)

CRITICAL QUESTIONS:
- Are these equations differentiable? (YES - all smooth functions)
- Do we need traffic model for all cities? (YES - key mechanism)
- Should we add stochastic shocks? (Not for 3-day scope)
- How do we validate forward simulation? (Test against historical data)

DESIGN DECISIONS:
- Traffic effects are OPTIONAL (via use_traffic_model flag)
- All functions are pure (no side effects) for JAX compatibility
- State is a dict (flexible for adding variables)
"""

import jax.numpy as jnp

def compute_emissions(state, action, params):
    """
    Compute emissions at time t
    
    E_t = A * exp(-η_τ*τ_t - η_c*c_t) * (1-κ*s_t) * (G_t/G_0)^γ * traffic_factor
    
    Args:
        state: dict with 'GDP', 'CO2', optional 'traffic_congestion'
        action: [τ, s, c] policy triple
        params: parameter dict (from config + city_config)
    
    Returns:
        float: Emissions in GtCO2/year
    
    QUESTIONS:
    - Should emission factor A scale with city size? (Yes - in city_config)
    - How do we handle negative emissions? (Can't happen with exp)
    - Should there be a floor on emissions? (Maybe - test sensitivity)
    """
    G_t = state['GDP']
    τ_t, s_t, c_t = action
    
    # Policy effects (exponential for carbon tax/charge, linear for subsidy)
    policy_effect = jnp.exp(-params['eta_tau'] * τ_t - params['eta_c'] * c_t)
    subsidy_effect = 1 - params['kappa'] * s_t
    
    # Economic driver (emissions scale with GDP)
    economic_driver = (G_t / params['G_0']) ** params['gamma']
    
    # Base emissions
    base_emissions = (params['A'] * 
                     policy_effect * 
                     subsidy_effect * 
                     economic_driver)
    
    # Traffic adjustment (if model enabled)
    if params.get('use_traffic_model', False) and 'traffic_congestion' in state:
        # Congestion increases emissions (slower speeds = more fuel per mile)
        congestion_multiplier = 1 + params['congestion_emission_factor'] * state['traffic_congestion']
        
        # Fleet composition (EVs reduce emissions)
        if 'ev_share' in state:
            # Weighted average: gas_share * emission_factor + ev_share * 0
            fleet_multiplier = 1 - state['ev_share']  # Simple: 1-EV% = gas%
        else:
            fleet_multiplier = 1.0
        
        E_t = base_emissions * congestion_multiplier * fleet_multiplier
    else:
        E_t = base_emissions
    
    return E_t


def update_co2(state, E_t, params):
    """
    Update atmospheric CO2 concentration
    
    CO2_{t+1} = CO2_t + α*E_t - δ*CO2_t
    
    This is a simple box model:
    - α: conversion from emissions to concentration
    - δ: natural removal (ocean/land sinks)
    
    QUESTIONS:
    - Should we use a more complex carbon cycle? (Not for 3 days)
    - Is linear removal realistic? (First approximation - good enough)
    - Should δ depend on CO2 level? (Could, but adds complexity)
    """
    CO2_t = state['CO2']
    
    CO2_next = CO2_t + params['alpha'] * E_t - params['delta'] * CO2_t
    
    return CO2_next


def update_gdp(state, action, params):
    """
    Update GDP with policy costs/benefits and climate damage
    
    G_{t+1} = G_t * (1+g_0) * (1+β_s*s) * (1-β_τ*τ) * (1-β_c*c) * 
              (1-θ*(CO2/CO2_0)^φ) * mobility_factor
    
    COMPONENTS:
    - (1+g_0): Baseline growth
    - (1+β_s*s): Subsidy benefit (transit → productivity)
    - (1-β_τ*τ): Carbon tax cost (energy prices up → growth down)
    - (1-β_c*c): Congestion charge cost
    - (1-θ*(CO2/CO2_0)^φ): Climate damage (DICE model)
    - mobility_factor: Congestion cost (from traffic model)
    
    QUESTIONS:
    - Should subsidy have diminishing returns? (Maybe β_s*s - 0.5*β_s*s^2)
    - Can climate damage exceed 100%? (No - we clip implicitly)
    - Should there be interaction terms? (Advanced - skip for now)
    """
    G_t = state['GDP']
    CO2_t = state['CO2']
    τ_t, s_t, c_t = action
    
    # Baseline growth
    baseline_growth = 1 + params['g_0']
    
    # Policy effects on GDP
    subsidy_benefit = 1 + params['beta_s'] * s_t
    carbon_tax_cost = 1 - params['beta_tau'] * τ_t
    congestion_charge_cost = 1 - params['beta_c'] * c_t
    
    # Climate damage (quadratic in CO2 increase)
    climate_damage = 1 - params['theta'] * (CO2_t / params['CO2_0']) ** params['phi']
    
    # Traffic mobility impact
    if params.get('use_traffic_model', False) and 'traffic_congestion' in state:
        # Congestion reduces productivity (wasted time, delayed freight)
        congestion_cost = params['mobility_elasticity'] * state['traffic_congestion']
        mobility_factor = 1 - congestion_cost
    else:
        mobility_factor = 1.0
    
    # Combine all factors
    G_next = (G_t * 
              baseline_growth * 
              subsidy_benefit * 
              carbon_tax_cost * 
              congestion_charge_cost * 
              climate_damage *
              mobility_factor)
    
    return G_next


def compute_traffic_metrics(state, action, params):
    """
    Compute traffic flow, congestion, and fleet composition
    
    This is a SIMPLIFIED traffic model. For detailed microsimulation,
    we'd use SUMO (future work).
    
    FLOW:
    1. Policies → Traffic volume reduction
    2. Volume + Capacity → Congestion (BPR function)
    3. Congestion → Speed reduction
    4. Policies → EV adoption
    
    QUESTIONS:
    - Is BPR function accurate enough? (Standard in transport planning)
    - Should we model peak vs. off-peak? (Not yet - daily average)
    - How does transit modal shift affect car traffic? (Via eta_s_traffic)
    """
    τ, s, c = action
    
    # ══════════════════════════════════════════════════════════
    # 1. BASE TRAFFIC VOLUME (scales with GDP)
    # ══════════════════════════════════════════════════════════
    base_traffic = params['base_traffic_volume']
    gdp_elasticity = params['gdp_traffic_elasticity']
    
    # Traffic grows with economy
    traffic_from_gdp = base_traffic * (state['GDP'] / params['GDP_baseline']) ** gdp_elasticity
    
    # ══════════════════════════════════════════════════════════
    # 2. POLICY EFFECTS ON VOLUME
    # ══════════════════════════════════════════════════════════
    # Carbon tax → people drive less
    tax_reduction = params['eta_tau_traffic'] * τ
    
    # Subsidies → shift to public transit
    subsidy_reduction = params['eta_s_traffic'] * s
    
    # Congestion charge → avoid driving
    charge_reduction = params['eta_c_traffic'] * c
    
    total_reduction = tax_reduction + subsidy_reduction + charge_reduction
    traffic_volume = traffic_from_gdp * (1 - total_reduction)
    
    # ══════════════════════════════════════════════════════════
    # 3. CONGESTION (BPR function)
    # ══════════════════════════════════════════════════════════
    road_capacity = params['road_capacity']
    v_c_ratio = traffic_volume / road_capacity
    
    # Bureau of Public Roads (BPR) congestion function
    # delay = free_flow_time * [1 + α*(V/C)^β]
    alpha_bpr = params['bpr_alpha']
    beta_bpr = params['bpr_beta']
    
    congestion_multiplier = 1 + alpha_bpr * (v_c_ratio ** beta_bpr)
    
    # Normalized congestion level (0 = free flow, 1 = at capacity)
    congestion_level = jnp.clip(v_c_ratio, 0, 1)
    
    # ══════════════════════════════════════════════════════════
    # 4. SPEED
    # ══════════════════════════════════════════════════════════
    free_flow_speed = params['free_flow_speed']
    actual_speed = free_flow_speed / congestion_multiplier
    
    # ══════════════════════════════════════════════════════════
    # 5. FLEET COMPOSITION (EV adoption)
    # ══════════════════════════════════════════════════════════
    base_ev_share = params['base_ev_share']
    ev_sensitivity = params['ev_tax_sensitivity']
    
    # Carbon tax accelerates EV adoption
    ev_share = base_ev_share + ev_sensitivity * τ
    ev_share = jnp.clip(ev_share, 0, 0.95)  # Max 95% EVs
    
    return {
        'traffic_volume': traffic_volume,
        'congestion': congestion_level,
        'speed': actual_speed,
        'ev_share': ev_share,
        'congestion_multiplier': congestion_multiplier
    }


def dynamics_step(state, action, params):
    """
    One timestep of the full coupled system
    
    ORDER MATTERS:
    1. Compute traffic (depends on current GDP, policies)
    2. Compute emissions (depends on GDP, traffic)
    3. Update CO2 (depends on emissions)
    4. Update GDP (depends on policies, CO2, traffic)
    
    Args:
        state: Current state dict
        action: Policy triple [τ, s, c]
        params: Full parameter dict (global + city-specific)
    
    Returns:
        next_state: Updated state dict
    
    QUESTIONS:
    - Should we use simultaneous vs. sequential updates? (Sequential is standard)
    - Do we need to store full history? (No - just current state for optimization)
    - Should we add noise/shocks? (Not for deterministic optimization)
    """
    
    # Traffic metrics (if enabled)
    if params.get('use_traffic_model', False):
        traffic = compute_traffic_metrics(state, action, params)
        # Add traffic state for emissions/GDP calculations
        state_with_traffic = {
            **state,
            'traffic_congestion': traffic['congestion'],
            'ev_share': traffic['ev_share']
        }
    else:
        traffic = {}
        state_with_traffic = state
    
    # Emissions
    E_t = compute_emissions(state_with_traffic, action, params)
    
    # CO2 accumulation
    CO2_next = update_co2(state_with_traffic, E_t, params)
    
    # GDP evolution
    G_next = update_gdp(state_with_traffic, action, params)
    
    # Build next state
    next_state = {
        'CO2': CO2_next,
        'GDP': G_next,
        'emissions': E_t,
        't': state['t'] + 1
    }
    
    # Add traffic metrics if available
    if traffic:
        next_state.update({
            'traffic_volume': traffic['traffic_volume'],
            'traffic_congestion': traffic['congestion'],
            'traffic_speed': traffic['speed'],
            'ev_share': traffic['ev_share']
        })
    
    return next_state


def simulate_trajectory(initial_state, policies, params):
    """
    Simulate full T-year trajectory
    
    This is the FORWARD MODEL that gets differentiated through.
    Must be:
    - Pure (no side effects)
    - Smooth (differentiable everywhere)
    - Deterministic (same input → same output)
    
    Args:
        initial_state: dict with CO2_0, G_0, t=0
        policies: array [T, 3] of [τ, s, c] for each timestep
        params: full parameter dict
    
    Returns:
        list of state dicts (length T+1, includes initial)
    
    QUESTIONS:
    - Should we use jax.lax.scan for efficiency? (Yes - but keep it simple first)
    - Do we need to validate policies in [0,1]? (Projection layer handles this)
    - Should we return trajectory as dict-of-arrays or list-of-dicts? (List for flexibility)
    """
    trajectory = [initial_state]
    state = initial_state
    
    T = len(policies)
    for t in range(T):
        state = dynamics_step(state, policies[t], params)
        trajectory.append(state)
    
    return trajectory