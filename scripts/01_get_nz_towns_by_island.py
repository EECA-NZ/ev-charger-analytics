import requests
import pandas as pd
import os
from constants import DATADIR, SOUTH_ISLAND_POLYGON, GEONAMES_USERNAME, POPULATION_THRESHOLD
from shapely.geometry import Point
import time

from helpers import extract_script_number

# Constants

NORTH_ISLAND_CSV_OUT = 'north_island_cities.csv'
SOUTH_ISLAND_CSV_OUT = 'south_island_cities.csv'

# Functions

def fetch_cities(username, population_threshold=1000):
    """
    Get population centres above population_threshold
    """
    url = f"http://api.geonames.org/searchJSON?country=NZ&maxRows=1000&username={username}"
    response = requests.get(url)
    data = response.json()
    cities = [
        city for city in data['geonames']
        if city.get('population', 0) > population_threshold
        and city.get('fcl', '') == 'P'  # Populated place
    ]
    return cities

def determine_island(city):
    """
    Determine which island each city is located on
    """
    point = Point(city['lng'], city['lat'])
    if SOUTH_ISLAND_POLYGON.contains(point):
        return "South"
    else:
        return "North"

# Main

start_time = time.time()

script_number = extract_script_number(__file__)

# Ensure DATADIR exists
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

cities = fetch_cities(GEONAMES_USERNAME, POPULATION_THRESHOLD)
df = pd.DataFrame(cities)[['name', 'lat', 'lng', 'population']]
df['island'] = df.apply(determine_island, axis=1)

# Save the results to separate CSV files for each island in the specified directory
df[df['island'] == 'North'].to_csv(f'{DATADIR}/{script_number}_{NORTH_ISLAND_CSV_OUT}', index=False)
df[df['island'] == 'South'].to_csv(f'{DATADIR}/{script_number}_{SOUTH_ISLAND_CSV_OUT}', index=False)

print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")