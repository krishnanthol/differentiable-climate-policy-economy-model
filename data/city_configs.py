"""
City-specific configuration parameters

This file defines the characteristics of each city we're studying, which for scope sake will be New York City and Los Angeles.
Each city has different:
- Economic scale (GDP)
- Traffic patterns (volume, capacity, speed)
- Infrastructure (transit availability, road network)
- Starting conditions (EV adoption, emissions)

KEY DESIGN DECISIONS:
1. We normalize CO2 to be the same (it's atmospheric - global)
2. GDP scales are realistic metro-area values
3. Traffic parameters reflect urban form (dense vs sprawl)
4. All parameters should be empirically justifiable

"""

import json
from pathlib import Path

# city archetypes 

CITY_CONFIGS= {
    # dense, traffic-congested, high transit NYC

    'nyc': {
        'name': 'New York City Metro',
        'archetype': 'dense_transit_rich',
        'description': 'High-density city with extensive transit but severe congestion',
        # Economic parameters
        'initial_conditions': {
            'CO2_0': 420.0,      # Atmospheric CO2 (same for all)
            'G_0': 1700.0,       # $1.7T metro GDP (2024 estimate)
            'E_0': 85.0          # 85 MtCO2/year (high - dense but large)
        },
        # Traffic characteristics
        'traffic_params': {
            'base_traffic_volume': 15000,   # vehicles/hour (very high)
            'road_capacity': 2500,          # vehicles/hour (limited - dense)
            'gdp_traffic_elasticity': 0.9,  # Traffic grows with economy
            'GDP_baseline': 1700.0,
            
            # BPR congestion function
            'bpr_alpha': 0.15,
            'bpr_beta': 4.0,
            'free_flow_speed': 45.0,        # mph (slow - urban traffic)
            
            # Fleet composition
            'base_ev_share': 0.08,          # 8% EVs (2024 estimate)
            'ev_tax_sensitivity': 0.4,      # Responsive to carbon tax
            
            # Emissions & mobility
            'congestion_emission_factor': 0.7,  # High impact of congestion
            'mobility_elasticity': 0.08,        # Congestion hurts economy more
            
            # Modal split (CRITICAL for transit-rich cities)
            'base_transit_share': 0.40,     # 40% already use transit
            'eta_tau_traffic': 0.25,        # Carbon tax → modest driving reduction
            'eta_s_traffic': 0.15,          # Subsidy → modest shift (already high)
            'eta_c_traffic': 0.50,          # Congestion charge → large effect!
        },
         # Economic sensitivity
        'economic_params': {
            'g_0': 0.025,           # 2.5% baseline growth
            'beta_s': 0.015,        # Subsidy has modest effect (transit exists)
            'beta_tau': 0.012,      # Carbon tax hurts (energy-intensive)
            'beta_c': 0.008,        # Congestion charge impacts economy
            'theta': 0.003,         # Climate damage coefficient
            'phi': 2.0,             # Damage non-linearity
        },
        
        # Policy constraints (optional city-specific bounds)
        'policy_bounds': {
            'tau_max': 1.0,
            's_max': 0.8,           # Less subsidy needed (transit exists)
            'c_max': 1.0,
        }

     },

     # second archetype: sprawling, car-dependent, highway-rich Los Angeles
     'la': {
        'name': 'Los Angeles Metro',
        'archetype': 'sprawl_car_dependent',
        'description': 'Low-density sprawl with extensive highways, car-dependent',
        
        'initial_conditions': {
            'CO2_0': 420.0,
            'G_0': 1000.0,       # $1.0T metro GDP
            'E_0': 70.0          # 70 MtCO2/year (car-dependent)
        },
        
        'traffic_params': {
            'base_traffic_volume': 12000,   # High but less than NYC
            'road_capacity': 3000,          # More highway capacity
            'gdp_traffic_elasticity': 1.0,  # Very car-dependent!
            'GDP_baseline': 1000.0,
            
            'bpr_alpha': 0.12,              # Less congestion (more roads)
            'bpr_beta': 3.5,
            'free_flow_speed': 60.0,        # Faster (highways)
            
            'base_ev_share': 0.12,          # 12% EVs (CA incentives)
            'ev_tax_sensitivity': 0.35,     # Moderate sensitivity
            
            'congestion_emission_factor': 0.5,  # Lower (better flow)
            'mobility_elasticity': 0.06,
            
            # Modal split
            'base_transit_share': 0.10,     # Only 10% transit!
            'eta_tau_traffic': 0.35,        # Carbon tax → strong effect
            'eta_s_traffic': 0.30,          # Subsidy → large shift potential
            'eta_c_traffic': 0.25,          # Congestion charge → moderate
        },
        
        'economic_params': {
            'g_0': 0.03,            # 3% growth (tech-driven)
            'beta_s': 0.025,        # Subsidy helps a lot (build transit)
            'beta_tau': 0.010,      # Carbon tax hurts less (cleaner grid)
            'beta_c': 0.006,        # Congestion charge hurts less
            'theta': 0.003,
            'phi': 2.0,
        },
        
        'policy_bounds': {
            'tau_max': 1.0,
            's_max': 1.0,           # Need more subsidy (no transit)
            'c_max': 0.8,           # Less effective (spread out)
        }
    }

}
# do some changes to find 2025 estimate numbers instead of 2024
# some helper functions
def load_city_config(city_name):
    """
    Loads city configuration for a specific city
    takes in argument city name, one of ['nyc', 'la']
    returns:
    dictionary of city parameters
    should raise valueerror if city not found
    """

    if city_name not in CITY_CONFIGS:
        available = list(CITY_CONFIGS.keys())
        raise ValueError(
            f"City '{city_name}' not found. "
            f"Available cities: {available}"
        )
    
    config = CITY_CONFIGS[city_name].copy()
    
    # validate required fields
    # Validate required fields
    required_fields = [
        'initial_conditions', 
        'traffic_params', 
        'economic_params',
        'policy_bounds'
    ]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"City config missing required field: {field}")
    
    return config

    # important to save a summary for city comparions
def get_city_summary():
    """
    Print summary of all cities for comparison
    
    QUESTIONS:
    - What summary stats are most useful?
    - Should we include derived metrics (e.g., emissions per capita)?
    """
    print("\n" + "="*70)
    print("CITY CONFIGURATIONS SUMMARY")
    print("="*70)
    
    for city_key, config in CITY_CONFIGS.items():
        print(f"\n{config['name']} ({city_key})")
        print(f"  Archetype: {config['archetype']}")
        print(f"  GDP: ${config['initial_conditions']['G_0']}B")
        print(f"  Emissions: {config['initial_conditions']['E_0']} MtCO2/year")
        print(f"  Emissions/GDP: {config['initial_conditions']['E_0']/config['initial_conditions']['G_0']:.3f}")
        print(f"  Traffic volume: {config['traffic_params']['base_traffic_volume']} veh/hr")
        print(f"  Road capacity: {config['traffic_params']['road_capacity']} veh/hr")
        print(f"  V/C ratio: {config['traffic_params']['base_traffic_volume']/config['traffic_params']['road_capacity']:.2f}")
        print(f"  Transit share: {config['traffic_params']['base_transit_share']*100:.0f}%")
        print(f"  EV share: {config['traffic_params']['base_ev_share']*100:.0f}%")


def compare_cities(metric_path):
    """
    Compare a specific metric across all cities
    
    Args:
        metric_path: Dot-notation path to metric, e.g. 
                    'traffic_params.base_ev_share'
    
    Example:
        compare_cities('traffic_params.base_ev_share')
        >>> NYC: 0.08, LA: 0.12
    """
    parts = metric_path.split('.')
    
    print(f"\nComparing: {metric_path}")
    print("-" * 50)
    
    for city_key, config in CITY_CONFIGS.items():
        value = config
        for part in parts:
            value = value[part]
        
        print(f"  {config['name']:25s}: {value}")

def save_configs_to_json(output_path='data/processed/city_configs.json'):
    """
    Save all city configs to JSON for easy access
    
    QUESTIONS:
    - Should we version these configs?
    - Should we track which config was used for each experiment?
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(CITY_CONFIGS, f, indent=2)
    
    print(f"✓ Saved city configs to: {output_path}")

# validate and test 
def validate_config(city_name):
    """
    Validate that city config is internally consistent
    
    QUESTIONS:
    - What consistency checks matter?
    - Should we check if parameters are in realistic ranges?
    """
    config = load_city_config(city_name)
    
    issues = []
    
    # Check 1: V/C ratio should be reasonable
    traffic = config['traffic_params']
    v_c_ratio = traffic['base_traffic_volume'] / traffic['road_capacity']
    if v_c_ratio > 10:
        issues.append(f"V/C ratio too high: {v_c_ratio:.1f}")
    
    # Check 2: Transit + driving should be <= 100%
    if traffic.get('base_transit_share', 0) > 1.0:
        issues.append("Transit share > 100%")
    
    # Check 3: Growth rate should be positive
    if config['economic_params']['g_0'] <= 0:
        issues.append("Negative growth rate")

    
    # Check 4: Policy sensitivities should be reasonable
    if traffic['eta_tau_traffic'] > 1.0:
        issues.append("Carbon tax traffic sensitivity too high")
    
    if issues:
        print(f"\n⚠️  Validation issues for {city_name}:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"✓ {city_name} config valid")
        return True

if __name__ == "__main__":
    """
    Run this file directly to:
    1. See summary of all cities
    2. Validate configurations
    3. Export to JSON
    """
    # Show summary
    get_city_summary()
    
    # Validate all
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    for city in CITY_CONFIGS.keys():
        validate_config(city)
    
    # Save to JSON
    save_configs_to_json()
    
    # Example comparisons
    print("\n" + "="*70)
    print("KEY COMPARISONS")
    print("="*70)
    compare_cities('traffic_params.base_ev_share')
    compare_cities('traffic_params.base_transit_share')
    compare_cities('economic_params.g_0')


# things to further improve on: review census data and acdaemic papers to verify these parameters 
# should transit share + car shre = 100%? most likely not but for sake of project, simplifying even though should include walking and biking