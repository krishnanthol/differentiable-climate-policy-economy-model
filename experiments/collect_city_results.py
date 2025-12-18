"""
Collect and analyze results from multi-city experiments

This script:
1. Scans output directory for completed experiments
2. Loads all summary.json files
3. Creates comparative analysis
4. Generates visualizations

USAGE:
    python experiments/collect_city_results.py \
        --job_id 12345 \
        --output_dir results/analysis

CRITICAL QUESTIONS:
- How do we handle failed experiments? (Skip or retry)
- Should we normalize metrics across cities? (Yes for comparison)
- What statistical tests should we use? (ANOVA for city differences)
- How do we identify outliers? (Z-score > 3)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats

sns.set_style('whitegrid')
sns.set_palette('husl')

def parse_args():
    parser = argparse.ArgumentParser(description='Collect and analyze multi-city results')
    parser.add_argument('--job_id', type=str, default=None,
                       help='SLURM job array ID (if not provided, uses latest)')
    parser.add_argument('--base_dir', type=str, 
                       default='results/multi_city',
                       help='Base results directory')
    parser.add_argument('--output_dir', type=str,
                       default='results/analysis',
                       help='Where to save analysis outputs')
    return parser.parse_args()


def find_latest_job_id(base_dir):
    """
    Find most recent job ID in results directory
    
    QUESTIONS:
    - Should we allow multiple job IDs? (Yes for meta-analysis)
    - How do we handle incomplete jobs? (Check COMPLETED marker)
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise ValueError(f"Base directory not found: {base_dir}")
    
    # Find all job directories (numeric names)
    job_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not job_dirs:
        raise ValueError(f"No job directories found in {base_dir}")
    
    # Sort by numeric ID, get latest
    latest = max(job_dirs, key=lambda x: int(x.name))
    
    print(f"Found latest job ID: {latest.name}")
    return latest.name


def collect_results(job_id, base_dir='results/multi_city'):
    """
    Collect all experiment results from a job
    
    Returns:
        DataFrame with one row per experiment
    
    QUESTIONS:
    - What if some experiments failed? (Skip, log warning)
    - Should we include training metrics? (Yes - convergence analysis)
    - How do we handle version differences? (Schema validation)
    """
    print("\n" + "="*70)
    print("COLLECTING RESULTS")
    print("="*70)
    
    base_path = Path(base_dir) / job_id
    
    if not base_path.exists():
        raise ValueError(f"Job directory not found: {base_path}")
    
    all_results = []
    failed_experiments = []
    
    # Scan all city/config combinations
    for city_dir in base_path.iterdir():
        if not city_dir.is_dir():
            continue
        
        city_name = city_dir.name
        
        for config_dir in city_dir.iterdir():
            if not config_dir.is_dir():
                continue
            
            config_name = config_dir.name
            summary_file = config_dir / 'summary.json'
            
            # Check for completion
            if (config_dir / 'FAILED').exists():
                failed_experiments.append(f"{city_name}/{config_name}")
                print(f"  ✗ Failed: {city_name}/{config_name}")
                continue
            
            if not summary_file.exists():
                print(f"  ⚠ Missing summary: {city_name}/{config_name}")
                continue
            
            # Load summary
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                
                # Extract key metrics
                result = {
                    # Experiment metadata
                    'city': city_name,
                    'city_name': summary['experiment']['city_name'],
                    'archetype': summary['experiment']['archetype'],
                    'config': config_name,
                    'job_id': job_id,
                    
                    # Configuration
                    'w_E': summary['config']['w_E'],
                    'w_G': summary['config']['w_G'],
                    'lambda': summary['config']['lambda_term'],
                    'learning_rate': summary['config']['lr'],
                    'num_iters': summary['config']['num_iters'],
                    
                    # City parameters
                    'initial_GDP': summary['city_params']['initial_GDP'],
                    'initial_emissions': summary['city_params']['initial_emissions'],
                    'initial_CO2': summary['city_params']['initial_CO2'],
                    'base_traffic': summary['city_params']['base_traffic'],
                    'base_transit_share': summary['city_params']['base_transit_share'],
                    'base_ev_share': summary['city_params']['base_ev_share'],
                    
                    # Training metrics
                    'iterations_completed': summary['training']['iterations_completed'],
                    'final_loss': summary['training']['final_loss'],
                    'best_loss': summary['training']['best_loss'],
                    'improvement_pct': summary['training']['improvement_pct'],
                    
                    # Economic outcomes
                    'final_GDP': summary['results']['final_GDP'],
                    'gdp_growth_pct': summary['results']['gdp_growth_pct'],
                    
                    # Climate outcomes
                    'final_CO2': summary['results']['final_CO2'],
                    'co2_change_pct': summary['results']['co2_change_pct'],
                    'total_emissions': summary['results']['total_emissions'],
                    'avg_emissions': summary['results']['avg_emissions'],
                    'emissions_reduction_pct': summary['results']['emissions_reduction_pct'],
                    
                    # Policies
                    'avg_tau': summary['results']['avg_policies']['tau'],
                    'avg_s': summary['results']['avg_policies']['s'],
                    'avg_c': summary['results']['avg_policies']['c'],
                    'final_tau': summary['results']['final_policies']['tau'],
                    'final_s': summary['results']['final_policies']['s'],
                    'final_c': summary['results']['final_policies']['c'],
                    
                    # Loss breakdown
                    'emissions_cost': summary['loss_breakdown']['emissions_cost'],
                    'gdp_cost': summary['loss_breakdown']['gdp_cost'],
                    'terminal_cost': summary['loss_breakdown']['terminal_cost'],
                }
                
                # Add traffic metrics if available
                if 'final_ev_share' in summary['results']:
                    result['final_ev_share'] = summary['results']['final_ev_share']
                    result['ev_adoption_pct'] = summary['results']['ev_adoption_pct']
                    result['avg_congestion'] = summary['results']['avg_congestion']
                
                all_results.append(result)
                print(f"  ✓ Loaded: {city_name}/{config_name}")
                
            except Exception as e:
                print(f"  ✗ Error loading {city_name}/{config_name}: {e}")
                failed_experiments.append(f"{city_name}/{config_name}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*70}")
    print(f"Collection Summary:")
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Failed experiments: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp in failed_experiments:
            print(f"    - {exp}")
    
    # Compute derived metrics
    if not df.empty:
        # Emissions intensity (emissions per GDP)
        df['emissions_per_gdp'] = df['total_emissions'] / df['final_GDP']
        
        # Policy mix (sum of all policies)
        df['policy_intensity'] = df['avg_tau'] + df['avg_s'] + df['avg_c']
        
        # Economic efficiency (GDP growth per unit policy)
        df['economic_efficiency'] = df['gdp_growth_pct'] / (df['policy_intensity'] + 0.01)
        
        # Climate efficiency (emissions reduction per unit policy)
        df['climate_efficiency'] = df['emissions_reduction_pct'] / (df['policy_intensity'] + 0.01)
    
    return df


def analyze_by_city(df):
    """
    Analyze results grouped by city
    
    STATISTICAL QUESTIONS:
    - Are differences between cities significant? (Use ANOVA)
    - Which city characteristics correlate with outcomes? (Regression)
    - How robust are results across configs? (Coefficient of variation)
    """
    print("\n" + "="*70)
    print("CITY-LEVEL ANALYSIS")
    print("="*70)
    
    # Group by city
    city_stats = df.groupby('city_name').agg({
        # Economic outcomes
        'gdp_growth_pct': ['mean', 'std', 'min', 'max'],
        'final_GDP': 'mean',
        
        # Climate outcomes
        'emissions_reduction_pct': ['mean', 'std', 'min', 'max'],
        'final_CO2': 'mean',
        'emissions_per_gdp': 'mean',
        
        # Policies
        'avg_tau': 'mean',
        'avg_s': 'mean',
        'avg_c': 'mean',
        'policy_intensity': 'mean',
        
        # Efficiency
        'economic_efficiency': 'mean',
        'climate_efficiency': 'mean',
    }).round(3)
    
    print("\nCity Statistics (averaged across configurations):")
    print(city_stats)
    
    # Test for significant differences between cities
    print("\n" + "-"*70)
    print("Statistical Tests (ANOVA):")
    print("-"*70)
    
    # Test GDP growth differences
    city_groups_gdp = [group['gdp_growth_pct'].values 
                       for name, group in df.groupby('city_name')]
    f_stat_gdp, p_val_gdp = stats.f_oneway(*city_groups_gdp)
    print(f"GDP Growth: F={f_stat_gdp:.2f}, p={p_val_gdp:.4f}", end='')
    print(" ***" if p_val_gdp < 0.001 else " **" if p_val_gdp < 0.01 else " *" if p_val_gdp < 0.05 else "")
    
    # Test emissions reduction differences
    city_groups_emissions = [group['emissions_reduction_pct'].values 
                            for name, group in df.groupby('city_name')]
    f_stat_em, p_val_em = stats.f_oneway(*city_groups_emissions)
    print(f"Emissions Reduction: F={f_stat_em:.2f}, p={p_val_em:.4f}", end='')
    print(" ***" if p_val_em < 0.001 else " **" if p_val_em < 0.01 else " *" if p_val_em < 0.05 else "")
    
    # Test policy differences
    city_groups_tau = [group['avg_tau'].values for name, group in df.groupby('city_name')]
    f_stat_tau, p_val_tau = stats.f_oneway(*city_groups_tau)
    print(f"Carbon Tax: F={f_stat_tau:.2f}, p={p_val_tau:.4f}", end='')
    print(" ***" if p_val_tau < 0.001 else " **" if p_val_tau < 0.01 else " *" if p_val_tau < 0.05 else "")
    
    return city_stats


def analyze_by_config(df):
    """
    Analyze results grouped by configuration
    
    QUESTIONS:
    - Which loss weights work best overall? (Compare across cities)
    - Are optimal configs consistent across cities? (Or city-specific?)
    - What's the emissions-GDP trade-off? (Pareto frontier)
    """
    print("\n" + "="*70)
    print("CONFIGURATION ANALYSIS")
    print("="*70)
    
    config_stats = df.groupby('config').agg({
        'final_loss': 'mean',
        'gdp_growth_pct': 'mean',
        'emissions_reduction_pct': 'mean',
        'avg_tau': 'mean',
        'avg_s': 'mean',
        'avg_c': 'mean',
    }).round(3)
    
    print("\nConfiguration Statistics (averaged across cities):")
    print(config_stats)
    
    # Find best config per city
    print("\n" + "-"*70)
    print("Best Configuration Per City:")
    print("-"*70)
    
    for city in df['city_name'].unique():
        city_data = df[df['city_name'] == city]
        
        # Best by lowest loss
        best = city_data.loc[city_data['final_loss'].idxmin()]
        
        print(f"\n{city}:")
        print(f"  Best config: {best['config']}")
        print(f"  Final loss: {best['final_loss']:.2f}")
        print(f"  GDP growth: +{best['gdp_growth_pct']:.1f}%")
        print(f"  Emissions reduction: {best['emissions_reduction_pct']:.1f}%")
        print(f"  Policies: τ={best['avg_tau']:.2f}, s={best['avg_s']:.2f}, c={best['avg_c']:.2f}")


def identify_patterns(df):
    """
    Identify cross-city patterns and insights
    
    RESEARCH QUESTIONS:
    - Do certain city types need certain policies?
    - Does existing transit affect optimal subsidy?
    - Does car-dependency affect carbon tax effectiveness?
    """
    print("\n" + "="*70)
    print("PATTERN IDENTIFICATION")
    print("="*70)
    
    # Pattern 1: Transit availability vs. subsidy need
    print("\nPattern 1: Transit vs. Subsidy")
    print("-" * 70)
    
    transit_subsidy = df.groupby('city_name').agg({
        'base_transit_share': 'first',
        'avg_s': 'mean'
    }).sort_values('base_transit_share')
    
    print(transit_subsidy)
    
    # Correlation
    corr = df['base_transit_share'].corr(df['avg_s'])
    print(f"\nCorrelation (transit share vs. subsidy): {corr:.3f}")
    
    if corr < -0.3:
        print("→ Cities with MORE transit need LESS subsidy (negative correlation)")
    elif corr > 0.3:
        print("→ Cities with MORE transit need MORE subsidy (positive correlation)")
    else:
        print("→ No strong relationship")
    
    # Pattern 2: City size vs. congestion charge
    print("\n\nPattern 2: City Size vs. Congestion Pricing")
    print("-" * 70)
    
    size_congestion = df.groupby('city_name').agg({
        'initial_GDP': 'first',
        'avg_c': 'mean'
    }).sort_values('initial_GDP', ascending=False)
    
    print(size_congestion)
    
    corr = df['initial_GDP'].corr(df['avg_c'])
    print(f"\nCorrelation (GDP vs. congestion charge): {corr:.3f}")
    
    if corr > 0.3:
        print("→ LARGER cities need MORE congestion pricing")
    
    # Pattern 3: Policy effectiveness by archetype
    print("\n\nPattern 3: Policy Effectiveness by Archetype")
    print("-" * 70)
    
    archetype_effectiveness = df.groupby('archetype').agg({
        'avg_tau': 'mean',
        'emissions_reduction_pct': 'mean',
        'climate_efficiency': 'mean'
    }).round(2)
    
    print(archetype_effectiveness)
    
    return {
        'transit_subsidy_correlation': corr,
        'patterns': archetype_effectiveness
    }


def create_visualizations(df, output_dir):
    """
    Generate comprehensive visualizations
    
    PLOTS TO CREATE:
    1. Policy mix by city (bar chart)
    2. Trade-off frontier (emissions vs GDP, by city)
    3. Box plots (outcomes by city)
    4. Heatmap (policy effectiveness)
    5. Correlation matrix (city characteristics vs outcomes)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # ══════════════════════════════════════════════════════════
    # FIGURE 1: Comprehensive Overview (2x3 subplots)
    # ══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Policy Mix by City
    policy_data = df.groupby('city_name')[['avg_tau', 'avg_s', 'avg_c']].mean()
    policy_data.plot(kind='bar', ax=axes[0, 0], width=0.8, rot=45)
    axes[0, 0].set_title('Average Policy Mix by City', fontweight='bold', fontsize=13)
    axes[0, 0].set_ylabel('Policy Value (0-1)')
    axes[0, 0].legend(['Carbon Tax (τ)', 'Transit Subsidy (s)', 'Congestion Charge (c)'])
    axes[0, 0].grid(alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1])
    
    # Plot 2: GDP Growth Distribution
    sns.boxplot(data=df, x='city_name', y='gdp_growth_pct', ax=axes[0, 1])
    axes[0, 1].set_title('GDP Growth Distribution by City', fontweight='bold', fontsize=13)
    axes[0, 1].set_ylabel('GDP Growth (%)')
    axes[0, 1].set_xlabel('City')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Plot 3: Emissions Reduction Distribution
    sns.boxplot(data=df, x='city_name', y='emissions_reduction_pct', ax=axes[0, 2])
    axes[0, 2].set_title('Emissions Reduction by City', fontweight='bold', fontsize=13)
    axes[0, 2].set_ylabel('Emissions Reduction (%)')
    axes[0, 2].set_xlabel('City')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(alpha=0.3, axis='y')
    
    # Plot 4: Trade-off Frontier (Emissions vs GDP)
    for city in df['city_name'].unique():
        city_data = df[df['city_name'] == city]
        axes[1, 0].scatter(city_data['emissions_reduction_pct'], 
                          city_data['gdp_growth_pct'],
                          label=city, s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Emissions Reduction (%)', fontsize=11)
    axes[1, 0].set_ylabel('GDP Growth (%)', fontsize=11)
    axes[1, 0].set_title('Climate-Economy Trade-off by City', fontweight='bold', fontsize=13)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 5: Emissions Intensity
    emissions_intensity = df.groupby('city_name')['emissions_per_gdp'].mean().sort_values()
    axes[1, 1].barh(range(len(emissions_intensity)), emissions_intensity.values, color='coral')
    axes[1, 1].set_yticks(range(len(emissions_intensity)))
    axes[1, 1].set_yticklabels(emissions_intensity.index)
    axes[1, 1].set_xlabel('Emissions per GDP (GtCO2 / $T)')
    axes[1, 1].set_title('Emissions Intensity by City', fontweight='bold', fontsize=13)
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    # Plot 6: EV Adoption (if available)
    if 'final_ev_share' in df.columns:
        ev_data = df.groupby('city_name')[['base_ev_share', 'final_ev_share']].mean() * 100
        x = np.arange(len(ev_data))
        width = 0.35
        axes[1, 2].bar(x - width/2, ev_data['base_ev_share'], width, label='Initial', alpha=0.8)
        axes[1, 2].bar(x + width/2, ev_data['final_ev_share'], width, label='Final', alpha=0.8)
        axes[1, 2].set_title('EV Adoption by City', fontweight='bold', fontsize=13)
        axes[1, 2].set_ylabel('EV Share (%)')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(ev_data.index, rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3, axis='y')
    else:
        axes[1, 2].text(0.5, 0.5, 'EV data not available', 
                       ha='center', va='center', fontsize=12)
        axes[1, 2].set_xlim([0, 1])
        axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path / 'city_comparison_overview.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: city_comparison_overview.png")
    plt.close()
    
    # ══════════════════════════════════════════════════════════
    # FIGURE 2: Policy Heatmap
    # ══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create pivot tables for heatmaps
    for idx, policy in enumerate([('avg_tau', 'Carbon Tax'), 
                                  ('avg_s', 'Transit Subsidy'),
                                  ('avg_c', 'Congestion Charge')]):
        policy_col, policy_name = policy
        
        pivot = df.pivot_table(
            values=policy_col,
            index='city_name',
            columns='config',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=axes[idx], cbar_kws={'label': 'Policy Value'})
        axes[idx].set_title(f'{policy_name} by City & Config', fontweight='bold')
        axes[idx].set_xlabel('Configuration')
        axes[idx].set_ylabel('City')
    
    plt.tight_layout()
    plt.savefig(output_path / 'policy_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: policy_heatmaps.png")
    plt.close()
    
    # ══════════════════════════════════════════════════════════
    # FIGURE 3: Correlation Matrix (City Characteristics)
    # ══════════════════════════════════════════════════════════
    # Select relevant columns for correlation
    corr_cols = [
        'base_transit_share', 'base_ev_share', 'initial_GDP',
        'avg_tau', 'avg_s', 'avg_c',
        'gdp_growth_pct', 'emissions_reduction_pct'
    ]
    
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, vmin=-1, vmax=1, square=True,
               cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix: City Characteristics & Outcomes', 
             fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: correlation_matrix.png")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


def generate_report(df, output_dir):
    """
    Generate summary report in Markdown
    
    REPORT SECTIONS:
    1. Executive Summary
    2. Key Findings by City
    3. Policy Recommendations
    4. Statistical Analysis
    5. Visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'ANALYSIS_REPORT.md'
    
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)
    
    with open(report_file, 'w') as f:
        # Header
        f.write("# Multi-City Climate Policy Optimization: Analysis Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(df)}\n\n")
        f.write(f"**Cities Analyzed:** {', '.join(df['city_name'].unique())}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        # Best overall result
        best = df.loc[df['final_loss'].idxmin()]
        f.write(f"**Best Overall Result:**\n")
        f.write(f"- City: {best['city_name']}\n")
        f.write(f"- Configuration: {best['config']}\n")
        f.write(f"- GDP Growth: +{best['gdp_growth_pct']:.1f}%\n")
        f.write(f"- Emissions Reduction: {best['emissions_reduction_pct']:.1f}%\n")
        f.write(f"- Optimal Policies: τ={best['avg_tau']:.2f}, s={best['avg_s']:.2f}, c={best['avg_c']:.2f}\n\n")
        
        # Key findings
        f.write("**Key Findings:**\n\n")
        
        # Finding 1: City-specific policies
        city_policy_variance = df.groupby('city_name')[['avg_tau', 'avg_s', 'avg_c']].std().mean(axis=1)
        if city_policy_variance.std() > 0.1:
            f.write("1. **Policies are highly city-specific** - optimal policy mix varies significantly across cities\n")
        else:
            f.write("1. **Policies are robust across cities** - similar policy mix works well everywhere\n")
        
        # Finding 2: Trade-offs
        corr_gdp_emissions = df['gdp_growth_pct'].corr(df['emissions_reduction_pct'])
        if corr_gdp_emissions < -0.3:
            f.write(f"2. **Strong trade-off exists** between GDP growth and emissions reduction (r={corr_gdp_emissions:.2f})\n")
        else:
            f.write(f"2. **Win-win possible** - GDP growth and emissions reduction are compatible (r={corr_gdp_emissions:.2f})\n")
        
        # Finding 3: Transit vs subsidy
        corr_transit = df['base_transit_share'].corr(df['avg_s'])
        if corr_transit < -0.3:
            f.write(f"3. **Cities with existing transit need less subsidy** (r={corr_transit:.2f})\n")
        elif corr_transit > 0.3:
            f.write(f"3. **Cities with more transit still benefit from subsidy** (r={corr_transit:.2f})\n")
        
        f.write("\n---\n\n")
        
        # City-by-City Results
        f.write("## Results by City\n\n")
        
        for city in df['city_name'].unique():
            city_data = df[df['city_name'] == city]
            best_city = city_data.loc[city_data['final_loss'].idxmin()]
            
            f.write(f"### {city}\n\n")
            f.write(f"**Archetype:** {best_city['archetype']}\n\n")
            f.write(f"**Characteristics:**\n")
            f.write(f"- GDP: ${best_city['initial_GDP']:.1f}B\n")
            f.write(f"- Emissions: {best_city['initial_emissions']:.1f} MtCO2/yr\n")
            f.write(f"- Transit Share: {best_city['base_transit_share']*100:.0f}%\n")
            f.write(f"- EV Share: {best_city['base_ev_share']*100:.1f}%\n\n")
            
            f.write(f"**Best Configuration:** {best_city['config']}\n\n")
            f.write(f"**Outcomes:**\n")
            f.write(f"- GDP Growth: +{best_city['gdp_growth_pct']:.1f}%\n")
            f.write(f"- Emissions Reduction: {best_city['emissions_reduction_pct']:.1f}%\n")
            f.write(f"- Final CO2: {best_city['final_CO2']:.1f} ppm\n\n")
            
            f.write(f"**Optimal Policies:**\n")
            f.write(f"- Carbon Tax: τ = {best_city['avg_tau']:.3f}\n")
            f.write(f"- Transit Subsidy: s = {best_city['avg_s']:.3f}\n")
            f.write(f"- Congestion Charge: c = {best_city['avg_c']:.3f}\n\n")
            
            f.write("---\n\n")
        
        # Policy Recommendations
        f.write("## Policy Recommendations\n\n")
        
        # Group cities by archetype
        for archetype in df['archetype'].unique():
            archetype_data = df[df['archetype'] == archetype]
            avg_policies = archetype_data[['avg_tau', 'avg_s', 'avg_c']].mean()
            
            f.write(f"### {archetype.replace('_', ' ').title()}\n\n")
            f.write(f"**Recommended Policy Mix:**\n")
            f.write(f"- Carbon Tax: {avg_policies['avg_tau']:.2f}\n")
            f.write(f"- Transit Subsidy: {avg_policies['avg_s']:.2f}\n")
            f.write(f"- Congestion Charge: {avg_policies['avg_c']:.2f}\n\n")
            
            # Interpretation
            if avg_policies['avg_tau'] > 0.6:
                f.write("→ **High carbon tax** needed to shift behavior\n")
            if avg_policies['avg_s'] > 0.5:
                f.write("→ **Strong transit investment** required\n")
            if avg_policies['avg_c'] > 0.5:
                f.write("→ **Congestion pricing** is effective\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        
        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("![City Comparison](city_comparison_overview.png)\n\n")
        f.write("![Policy Heatmaps](policy_heatmaps.png)\n\n")
        f.write("![Correlation Matrix](correlation_matrix.png)\n\n")
        
        f.write("---\n\n")
        f.write("**End of Report**\n")
    
    print(f"✓ Report saved to: {report_file}")


def main():
    args = parse_args()
    
    # Find job ID if not provided
    if args.job_id is None:
        args.job_id = find_latest_job_id(args.base_dir)
    
    # Collect results
    df = collect_results(args.job_id, args.base_dir)
    
    if df.empty:
        print("\n✗ No results found!")
        return
    
    # Save raw data
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / 'all_results.csv', index=False)
    print(f"\n✓ Saved raw data to: {output_path / 'all_results.csv'}")
    
    # Run analyses
    city_stats = analyze_by_city(df)
    config_stats = analyze_by_config(df)
    patterns = identify_patterns(df)
    
    # Generate visualizations
    create_visualizations(df, args.output_dir)
    
    # Generate report
    generate_report(df, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review ANALYSIS_REPORT.md")
    print("  2. Examine visualizations")
    print("  3. Share findings with team")


if __name__ == "__main__":
    main()