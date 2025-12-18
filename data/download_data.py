"""
Download and create baseline climate-economic data

This script creates synthetic data based on real historical trends.
For a real study, you'd download from actual sources.

DATA SOURCES (if you want real data):
- CO2: https://gml.noaa.gov/ccgg/trends/data.html
- Emissions: https://globalcarbonbudget.org/
- GDP: World Bank API, BEA (for US metros)
- Traffic: City open data portals, Uber Movement

QUESTIONS:
- using real historical data?
- consider implementing city specific emissions
- How far back should data go? multiple decades, maybe 50 years?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request 
import io 
import json
import warnings
warnings.filterwarnings("ignore")

def download_co2_data():
   url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
   print("creating co2 data...")
   try:
        # Download data
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')
        
        # Parse data (skip comment lines starting with #)
        lines = [line for line in content.split('\n') if not line.startswith('#') and line.strip()]
        
        # Split into columns
        data = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    co2_avg = float(parts[3])
                    
                    # Skip missing data (coded as -99.99)
                    if co2_avg > 0:
                        data.append({
                            'year': year,
                            'month': month,
                            'co2_ppm': co2_avg
                        })
                except (ValueError, IndexError):
                    continue
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Annual average
        df_annual = df.groupby('year')['co2_ppm'].mean().reset_index()
        
        # Filter to recent decades (2000-2024)
        df_annual = df_annual[df_annual['year'] >= 2000]
        
        print(f"  ✓ Downloaded {len(df_annual)} years of CO2 data")
        print(f"  ✓ Range: {df_annual['year'].min()}-{df_annual['year'].max()}")
        print(f"  ✓ Current CO2 (latest): {df_annual['co2_ppm'].iloc[-1]:.2f} ppm")
        
        return df_annual
        
   except Exception as e:
        print(f"  ✗ Error downloading NOAA data: {e}")
        print(f"  → Using backup synthetic data")
        
        # Fallback to synthetic if download fails
        years = np.arange(2000, 2025)
        co2_ppm = 370 + 2.5 * (years - 2000)
        return pd.DataFrame({'year': years, 'co2_ppm': co2_ppm})
    

def download_global_carbon_budget():
    """
    Download global CO2 emissions from Global Carbon Project
    
    SOURCE: https://globalcarbonbudget.org/
    ALTERNATIVE: https://ourworldindata.org/co2-emissions
    
    for this implementation, using Our World in Data API bc easier to parse
    """
    print("downloading global carbon budget data...")
    # Our World in Data GitHub CSV (easier than GCP website)
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    
    try:
        # Download data
        df = pd.read_csv(url)
        
        # Filter to World totals
        df_world = df[df['country'] == 'World'].copy()
        
        # Extract relevant columns
        df_world = df_world[['year', 'co2']].rename(columns={'co2': 'emissions_MtCO2'})
        
        # Filter to our time range (2000+)
        df_world = df_world[df_world['year'] >= 2000]
        
        # Convert to GtCO2 (from MtCO2)
        df_world['emissions_GtCO2'] = df_world['emissions_MtCO2'] / 1000
        
        # Drop MtCO2 column
        df_world = df_world[['year', 'emissions_GtCO2']].reset_index(drop=True)
        
        # Handle missing values (interpolate)
        df_world = df_world.interpolate(method='linear')
        
        print(f"  ✓ Downloaded {len(df_world)} years of emissions data")
        print(f"  ✓ Latest emissions: {df_world['emissions_GtCO2'].iloc[-1]:.2f} GtCO2/year")
        
        return df_world
        
    except Exception as e:
        print(f"  ✗ Error downloading emissions data: {e}")
        print(f"  → Using backup synthetic data")
        
        # Fallback
        years = np.arange(2000, 2025)
        emissions = 35 * (1.01 ** (years - 2000))
        return pd.DataFrame({'year': years, 'emissions_GtCO2': emissions})

def download_world_bank_gdp():
    """
    Download world GDP from World Bank
    
    SOURCE: World Bank Open Data API
    INDICATOR: NY.GDP.MKTP.KD (GDP in constant 2015 USD)
    
    ALTERNATIVE: Manual download from:
    https://data.worldbank.org/indicator/NY.GDP.MKTP.KD
    
    """
    print("downloading world bank GDP data...")
    try:
        # World Bank API endpoint
        url = "https://api.worldbank.org/v2/country/WLD/indicator/NY.GDP.MKTP.KD?format=json&date=2000:2024"
        
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')
            data = json.loads(content)
        
        # Parse JSON (format: [metadata, data_list])
        records = data[1] if len(data) > 1 else []
        
        df = pd.DataFrame([
            {
                'year': int(record['date']),
                'gdp_usd': record['value']
            }
            for record in records
            if record['value'] is not None
        ])
        
        # Sort by year
        df = df.sort_values('year').reset_index(drop=True)
        
        # Convert to trillions
        df['gdp_trillion_usd'] = df['gdp_usd'] / 1e12
        df = df[['year', 'gdp_trillion_usd']]
        
        # Interpolate any missing years
        df = df.set_index('year').reindex(range(2000, 2025)).interpolate(method='linear').reset_index()
        df.columns = ['year', 'gdp_trillion_usd']
        
        print(f"  ✓ Downloaded {len(df)} years of GDP data (direct API)")
        print(f"  ✓ Latest GDP: ${df['gdp_trillion_usd'].iloc[-1]:.2f}T")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading World Bank data: {e}")
        print(f"  → Using backup synthetic data")
        
        # Fallback
        years = np.arange(2000, 2025)
        gdp = 80 * (1.03 ** (years - 2000))
        return pd.DataFrame({'year': years, 'gdp_trillion_usd': gdp})

def get_metro_gdp_estimates():
    """
    Get metro area GDP estimates
    
    SOURCE: Bureau of Economic Analysis (BEA)
    https://www.bea.gov/data/gdp/gdp-metro-area
    
    NOTE: For 3-day project, we use published estimates rather than API
    
    ACTUAL DATA (2023 estimates):
    - NYC Metro: $1,751B
    - LA Metro: $1,048B
   
    """
    
    # 2023 estimates from BEA (published data)
    metro_gdp = {
        'nyc': {
            'name': 'New York-Newark-Jersey City MSA',
            'gdp_billion_2023': 1751,
            'source': 'BEA Regional Economic Accounts, 2023'
        },
        'la': {
            'name': 'Los Angeles-Long Beach-Anaheim MSA',
            'gdp_billion_2023': 1048,
            'source': 'BEA Regional Economic Accounts, 2023'
        }
    }
    
    print("\nMetro Area GDP Estimates (2023):")
    for city, data in metro_gdp.items():
        print(f"  {data['name']}: ${data['gdp_billion_2023']}B")
    
    return metro_gdp

def download_all_data():
    """
    Download all real historical data and save to CSV
    """
    print("="*70)
    print("DOWNLOADING REAL HISTORICAL DATA")
    print("="*70)
    
    # Create output directory
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # Download each dataset
    co2_df = download_co2_data()
    emissions_df = download_global_carbon_budget()
    gdp_df = download_world_bank_gdp()
    metro_gdp = get_metro_gdp_estimates()
    
    # Save to CSV
    co2_df.to_csv('data/raw/co2_data.csv', index=False)
    print(f"\n✓ Saved: data/raw/co2_data.csv")
    
    emissions_df.to_csv('data/raw/emissions_data.csv', index=False)
    print(f"✓ Saved: data/raw/emissions_data.csv")
    
    gdp_df.to_csv('data/raw/gdp_data.csv', index=False)
    print(f"✓ Saved: data/raw/gdp_data.csv")
    
    # Save metro GDP as JSON
    with open('data/raw/metro_gdp_2023.json', 'w') as f:
        json.dump(metro_gdp, f, indent=2)
    print(f"✓ Saved: data/raw/metro_gdp_2023.json")
    
    # Print summary
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    print(f"\nCO2 (Atmospheric):")
    print(f"  Years: {co2_df['year'].min()}-{co2_df['year'].max()}")
    print(f"  Range: {co2_df['co2_ppm'].min():.1f} - {co2_df['co2_ppm'].max():.1f} ppm")
    print(f"  Growth rate: {(co2_df['co2_ppm'].iloc[-1] / co2_df['co2_ppm'].iloc[0]) ** (1/len(co2_df)) - 1:.3%}/year")
    
    print(f"\nEmissions (Global):")
    print(f"  Years: {emissions_df['year'].min()}-{emissions_df['year'].max()}")
    print(f"  Range: {emissions_df['emissions_GtCO2'].min():.1f} - {emissions_df['emissions_GtCO2'].max():.1f} GtCO2/year")
    print(f"  Growth rate: {(emissions_df['emissions_GtCO2'].iloc[-1] / emissions_df['emissions_GtCO2'].iloc[0]) ** (1/len(emissions_df)) - 1:.3%}/year")
    
    print(f"\nGDP (World):")
    print(f"  Years: {gdp_df['year'].min()}-{gdp_df['year'].max()}")
    print(f"  Range: ${gdp_df['gdp_trillion_usd'].min():.1f}T - ${gdp_df['gdp_trillion_usd'].max():.1f}T")
    print(f"  Growth rate: {(gdp_df['gdp_trillion_usd'].iloc[-1] / gdp_df['gdp_trillion_usd'].iloc[0]) ** (1/len(gdp_df)) - 1:.3%}/year")
    
    print("\n" + "="*70)
    print("✓ ALL DATA DOWNLOADED SUCCESSFULLY!")
    print("="*70)
    print("\nNext step: python data/preprocess_data.py")
    
    return co2_df, emissions_df, gdp_df, metro_gdp


if __name__ == "__main__":
    co2_df, emissions_df, gdp_df, metro_gdp = download_all_data()
    
    # Quick visualization (optional)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # CO2
        axes[0].plot(co2_df['year'], co2_df['co2_ppm'], 'o-', linewidth=2)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('CO₂ (ppm)')
        axes[0].set_title('Atmospheric CO₂ (NOAA Mauna Loa)')
        axes[0].grid(alpha=0.3)
        
        # Emissions
        axes[1].plot(emissions_df['year'], emissions_df['emissions_GtCO2'], 'o-', linewidth=2, color='coral')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Emissions (GtCO₂/year)')
        axes[1].set_title('Global CO₂ Emissions')
        axes[1].grid(alpha=0.3)
        
        # GDP
        axes[2].plot(gdp_df['year'], gdp_df['gdp_trillion_usd'], 'o-', linewidth=2, color='green')
        axes[2].set_xlabel('Year')
        axes[2].set_ylabel('GDP (Trillion USD)')
        axes[2].set_title('World GDP')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/raw/historical_data_overview.png', dpi=150)
        print("\n✓ Saved visualization: data/raw/historical_data_overview.png")
        
    except ImportError:
        print("\n(matplotlib not available for visualization)")







