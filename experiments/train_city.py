"""
Train city-specific climate policies

This is the MAIN TRAINING SCRIPT for multi-city optimization.

USAGE:
    python experiments/train_city.py \
        --city nyc \
        --w_E 1.0 --w_G 1.0 --lambda_term 10.0 \
        --lr 0.01 --num_iters 1000 \
        --output_dir results/nyc/balanced

WHAT IT DOES:
1. Load city-specific configuration
2. Merge with global parameters
3. Initialize policy network
4. Run gradient-based optimization
5. Save results (policies, trajectory, metrics)

CRITICAL QUESTIONS:
- Should we use different learning rates per city? (Could tune)
- Should we warm-start from similar city? (Advanced)
- How do we know if optimization converged? (Check gradient norm)
- Should we save checkpoints? (Yes for long runs)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
sys.path.append('.')

import jax
import jax.numpy as jnp
import optax
import numpy as np

# Import our modules
from data.city_configs import load_city_config
from src.dynamics import simulate_trajectory
from src.losses import compute_loss, compute_loss_components
from src.models import init_policy_params, get_trajectory_policies

def parse_args():
    """
    Parse command-line arguments
    
    DESIGN: Use argparse for flexibility
    - Required: city, output_dir, name
    - Optional: hyperparameters (with defaults)
    
    QUESTIONS:
    - Should we use config files instead? (Could, but CLI is more flexible)
    - Should we validate argument combinations? (Yes - add checks)
    """
    parser = argparse.ArgumentParser(
        description='Train climate-economic policy optimization for specific city',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # City selection
    parser.add_argument('--city', type=str, required=True,
                       choices=['nyc', 'la', 'college_park', 'baltimore'],
                       help='City to optimize policies for')
    
    # Loss weights
    parser.add_argument('--w_E', type=float, default=1.0,
                       help='Emissions penalty weight')
    parser.add_argument('--w_G', type=float, default=1.0,
                       help='GDP reward weight')
    parser.add_argument('--lambda_term', type=float, default=10.0,
                       help='Terminal CO2 penalty')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (Adam optimizer)')
    parser.add_argument('--num_iters', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--T', type=int, default=50,
                       help='Time horizon (years)')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--name', type=str, required=True,
                       help='Experiment name')
    
    # Optional flags
    parser.add_argument('--use_traffic', type=str, default='true',
                       choices=['true', 'false'],
                       help='Enable traffic model')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed logs')
    
    return parser.parse_args()


def setup_parameters(args):
    """
    Set up complete parameter dict by merging:
    1. Global parameters (from calibration)
    2. City-specific parameters
    3. Command-line overrides
    
    QUESTIONS:
    - What if parameters conflict? (City-specific overrides global)
    - Should we validate parameter ranges? (Yes - add checks)
    - How do we handle missing parameters? (Use defaults, raise error)
    """
    print("\n" + "="*60)
    print("PARAMETER SETUP")
    print("="*60)
    
    # Load global baseline parameters
    with open('data/processed/parameters.json') as f:
        global_params = json.load(f)
    
    # Load city-specific configuration
    city_config = load_city_config(args.city)
    
    print(f"\nCity: {city_config['name']}")
    print(f"  Archetype: {city_config['archetype']}")
    print(f"  Description: {city_config['description']}")
    
    # Merge parameters (city-specific overrides global)
    params = {**global_params}
    params.update(city_config['traffic_params'])
    params.update(city_config['economic_params'])
    
    # Add city-specific initial conditions
    params.update(city_config['initial_conditions'])
    
    # Set traffic model flag
    params['use_traffic_model'] = args.use_traffic.lower() == 'true'
    
    # Loss weights
    loss_weights = {
        'w_E': args.w_E,
        'w_G': args.w_G,
        'lambda': args.lambda_term
    }
    
    print(f"\nParameters loaded:")
    print(f"  GDP baseline: ${params['G_0']:.1f}B")
    print(f"  Emissions baseline: {params['E_0']:.1f} MtCO2/yr")
    print(f"  Traffic volume: {params['base_traffic_volume']} veh/hr")
    print(f"  Transit share: {params['base_transit_share']*100:.0f}%")
    print(f"  EV share: {params['base_ev_share']*100:.1f}%")
    
    print(f"\nLoss weights:")
    print(f"  w_E (emissions): {loss_weights['w_E']}")
    print(f"  w_G (GDP): {loss_weights['w_G']}")
    print(f"  λ (terminal CO2): {loss_weights['lambda']}")
    
    return params, loss_weights, city_config


def train(args):
    """
    Main training loop
    
    ALGORITHM:
    1. Initialize policy network randomly
    2. For each iteration:
        a. Generate policy sequence from network
        b. Simulate forward dynamics
        c. Compute loss
        d. Compute gradients via autodiff
        e. Update network parameters with Adam
    3. Save final results
    
    QUESTIONS:
    - Should we use different optimizer? (Adam works well)
    - Should we add gradient clipping? (Can help stability)
    - Should we use learning rate schedule? (Could anneal)
    - How do we detect convergence? (Track loss plateau)
    """
    print("\n" + "="*60)
    print(f"TRAINING: {args.name}")
    print("="*60)
    print(f"Device: {jax.devices()}")
    
    # Setup
    params, loss_weights, city_config = setup_parameters(args)
    
    # Initial state (from city config)
    initial_state = {
        'CO2': params['CO2_0'],
        'GDP': params['G_0'],
        't': 0
    }
    
    # State parameters for policy network normalization
    state_params = {
        'CO2_0': params['CO2_0'],
        'G_0': params['G_0']
    }
    
    print(f"\nInitial state:")
    print(f"  CO2: {initial_state['CO2']:.1f} ppm")
    print(f"  GDP: ${initial_state['GDP']:.1f}B")
    
    # Initialize policy network
    key = jax.random.PRNGKey(args.seed)
    policy_params = init_policy_params(key)
    
    # Optimizer (Adam)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(policy_params)
    
    print(f"\nTraining configuration:")
    print(f"  Time horizon: {args.T} years")
    print(f"  Learning rate: {args.lr}")
    print(f"  Iterations: {args.num_iters}")
    print(f"  Traffic model: {params['use_traffic_model']}")
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print("="*60)
    
    losses = []
    best_loss = float('inf')
    best_params = None
    
    for iteration in range(args.num_iters):
        
        # Define loss function for this iteration
        def loss_fn(policy_params):
            """
            Loss function to be minimized
            
            This gets differentiated by JAX!
            """
            # Generate policy sequence
            policies = get_trajectory_policies(
                policy_params, 
                initial_state, 
                args.T,
                state_params
            )
            
            # Simulate forward
            trajectory = simulate_trajectory(initial_state, policies, params)
            
            # Compute loss
            return compute_loss(trajectory, params, loss_weights)
        
        # Compute loss and gradients
        loss_value, grads = jax.value_and_grad(loss_fn)(policy_params)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        policy_params = optax.apply_updates(policy_params, updates)
        
        # Track loss
        losses.append(float(loss_value))
        
        # Save best
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = policy_params
        
        # Log progress
        if iteration % 100 == 0 or iteration == args.num_iters - 1:
            print(f"  Iter {iteration:4d}: Loss = {loss_value:8.2f} "
                  f"(best: {best_loss:8.2f})")
        
        # Early stopping (optional)
        if iteration > 200 and losses[-1] > losses[-100]:
            # Loss hasn't improved in 100 iters
            if args.verbose:
                print(f"\n  Early stopping at iteration {iteration}")
            break
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Final loss: {losses[-1]:.2f}")
    print(f"  Best loss: {best_loss:.2f}")
    print(f"  Improvement: {(losses[0] - best_loss)/losses[0]*100:.1f}%")
    
    # Use best parameters
    policy_params = best_params
    
    # ══════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SAVING RESULTS")
    print("="*60)
    print(f"Output directory: {output_path}")
    
    # Save policy parameters
    with open(output_path / 'policy_params.pkl', 'wb') as f:
        pickle.dump(policy_params, f)
    print(f"  ✓ Saved policy parameters")
    
    # Save training losses
    with open(output_path / 'losses.json', 'w') as f:
        json.dump(losses, f)
    print(f"  ✓ Saved training losses")
    
    # Get final trajectory
    final_policies = get_trajectory_policies(
        policy_params, initial_state, args.T, state_params
    )
    final_trajectory = simulate_trajectory(initial_state, final_policies, params)
    
    # Save policies
    np.save(output_path / 'policies.npy', final_policies)
    print(f"  ✓ Saved final policies")
    
    # Save trajectory
    trajectory_json = []
    for state in final_trajectory:
        state_json = {
            k: float(v) if not isinstance(v, (dict, int)) else v 
            for k, v in state.items()
        }
        trajectory_json.append(state_json)
    
    with open(output_path / 'trajectory.json', 'w') as f:
        json.dump(trajectory_json, f, indent=2)
    print(f"  ✓ Saved trajectory")
    
    # Compute detailed metrics
    loss_breakdown = compute_loss_components(final_trajectory, params, loss_weights)
    
    # Create summary
    summary = {
        'experiment': {
            'name': args.name,
            'city': args.city,
            'city_name': city_config['name'],
            'archetype': city_config['archetype'],
            'timestamp': str(pd.Timestamp.now())
        },
        'config': vars(args),
        'city_params': {
            'initial_GDP': params['G_0'],
            'initial_emissions': params['E_0'],
            'initial_CO2': params['CO2_0'],
            'base_traffic': params['base_traffic_volume'],
            'base_transit_share': params['base_transit_share'],
            'base_ev_share': params['base_ev_share']
        },
        'training': {
            'iterations_completed': len(losses),
            'final_loss': losses[-1],
            'best_loss': best_loss,
            'initial_loss': losses[0],
            'improvement_pct': (losses[0] - best_loss) / losses[0] * 100
        },
        'results': {
            'final_CO2': final_trajectory[-1]['CO2'],
            'final_GDP': final_trajectory[-1]['GDP'],
            'co2_change_pct': (final_trajectory[-1]['CO2'] / initial_state['CO2'] - 1) * 100,
            'gdp_growth_pct': (final_trajectory[-1]['GDP'] / initial_state['GDP'] - 1) * 100,
            'total_emissions': sum(s.get('emissions', 0) for s in final_trajectory[1:]),
            'avg_emissions': sum(s.get('emissions', 0) for s in final_trajectory[1:]) / args.T,
            'emissions_reduction_pct': (1 - sum(s.get('emissions', 0) for s in final_trajectory[1:]) / (args.T * params['E_0'])) * 100,
            'avg_policies': {
                'tau': float(jnp.mean(final_policies[:, 0])),
                's': float(jnp.mean(final_policies[:, 1])),
                'c': float(jnp.mean(final_policies[:, 2]))
            },
            'final_policies': {
                'tau': float(final_policies[-1, 0]),
                's': float(final_policies[-1, 1]),
                'c': float(final_policies[-1, 2])
            }
        },
        'loss_breakdown': loss_breakdown
    }
    
    # Add traffic metrics if available
    if params['use_traffic_model'] and 'ev_share' in final_trajectory[-1]:
        summary['results']['final_ev_share'] = final_trajectory[-1]['ev_share']
        summary['results']['ev_adoption_pct'] = (final_trajectory[-1]['ev_share'] - params['base_ev_share']) * 100
        summary['results']['avg_congestion'] = np.mean([s.get('traffic_congestion', 0) for s in final_trajectory[1:]])
    
    # Save summary
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved summary")
    
    # ══════════════════════════════════════════════════════════
    # PRINT RESULTS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nCity: {city_config['name']}")
    print(f"Experiment: {args.name}")
    
    print(f"\nEconomic Outcomes:")
    print(f"  Initial GDP: ${summary['results']['final_GDP']/(1 + summary['results']['gdp_growth_pct']/100):.1f}B")
    print(f"  Final GDP:   ${summary['results']['final_GDP']:.1f}B")
    print(f"  Growth:      +{summary['results']['gdp_growth_pct']:.1f}%")
    
    print(f"\nClimate Outcomes:")
    print(f"  Initial CO2: {initial_state['CO2']:.1f} ppm")
    print(f"  Final CO2:   {summary['results']['final_CO2']:.1f} ppm")
    print(f"  Change:      +{summary['results']['co2_change_pct']:.1f}%")
    print(f"  Total Emissions: {summary['results']['total_emissions']:.1f} GtCO2")
    print(f"  Reduction vs Baseline: {summary['results']['emissions_reduction_pct']:.1f}%")
    
    print(f"\nOptimal Policies (average over {args.T} years):")
    print(f"  Carbon Tax (τ):      {summary['results']['avg_policies']['tau']:.3f}")
    print(f"  Transit Subsidy (s): {summary['results']['avg_policies']['s']:.3f}")
    print(f"  Congestion Charge (c): {summary['results']['avg_policies']['c']:.3f}")
    
    if 'final_ev_share' in summary['results']:
        print(f"\nTransportation:")
        print(f"  Initial EV share: {params['base_ev_share']*100:.1f}%")
        print(f"  Final EV share:   {summary['results']['final_ev_share']*100:.1f}%")
        print(f"  Adoption increase: +{summary['results']['ev_adoption_pct']:.1f}%")
    
    print(f"\n{'='*60}")
    print("✓ EXPERIMENT COMPLETE!")
    print("="*60)
    
    return policy_params, final_policies, final_trajectory, summary


if __name__ == "__main__":
    import pandas as pd  # For timestamp
    
    args = parse_args()
    
    # Run training
    policy_params, policies, trajectory, summary = train(args)