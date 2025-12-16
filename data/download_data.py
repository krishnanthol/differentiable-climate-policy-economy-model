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

def download_co2_data():
    TODO()
   #  url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    # co2_dataframe = pd.read_csv(url, delim_whitespace=True, comment="#",
                                     # names=["year", "month", "decimal_date", "average", "interpolated", "trend", "days"])
    # print("creating co2 data...")
    #
    # years = np.arange(1975, 2025) # lets do 50 years
    # real trend data shows about 2.3 ppm increase per year recently
    # co2_values = 330 + 2.3 * (years - 1975) + np.random.normal(0, 1, size=len(years))
    """ 
        df = pd.DataFrame({
            'year': years,
            'co2_values': co2_values

        })
    """

 
    # data_dir = Path(__file__).parent / "data"

    # Path('data/raw').mkdir(parents=True, exist_ok=True)
    # df.to_csv(co2, index=False)
    # print(f"✓ CO2 data saved: {len(df)} years")

    # return co2_dataframe

def download_emissions_data():
    TODO()

