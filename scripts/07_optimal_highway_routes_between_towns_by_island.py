import os
import sys
import pandas as pd
import networkx as nx
from haversine import haversine
import pickle
import time

from helpers import extract_script_number, find_data_file

from constants import DATADIR, WHAT_TO_OPTIMIZE

# Constants

INPUT_NORTH_ISLAND_CITIES_FILENAME = 'north_island_cities.csv'
INPUT_SOUTH_ISLAND_CITIES_FILENAME = 'south_island_cities.csv'

INPUT_NORTH_ISLAND_ROADS_FILENAME = 'north_island_abley_roads_with_towns_and_ev_charging_stations.gpickle'
INPUT_SOUTH_ISLAND_ROADS_FILENAME = 'south_island_abley_roads_with_towns_and_ev_charging_stations.gpickle'


# Functions

def calculate_routes(island_roads, island_towns):
    """
    Precomputes shortest-path trees for each city, then uses them to calculate
    the shortest path between each pair of cities. In this approach, the time
    complexity for the shortest path computation is
        O(n*mlogm), where
        n is the number of towns and
        m is the number of nodes in the graph.
    This is more efficient than the naive all-pairs shortest path method because
        n<<m.
    """
    shortest_path_trees = {}
    total_cities = len(island_towns)
    print("Precompute shortest path trees for each city")
    for idx, city in enumerate(island_towns['name']):
        percent_complete = idx / total_cities * 100
        sys.stdout.write(f"\r{percent_complete:.2f}% complete, working on city {city}" + " " * 50)
        sys.stdout.flush()
        shortest_path_trees[city] = nx.single_source_dijkstra_path(island_roads, source=city, weight=WHAT_TO_OPTIMIZE)
    print()
    routes = {}
    for idx1, city1 in enumerate(island_towns['name']):
        for idx2, city2 in enumerate(island_towns['name']):
            percent_complete = (idx1 * total_cities + idx2) / (total_cities * total_cities) * 100
            sys.stdout.write(f"\r{percent_complete:.2f}% complete, working on route {city1} to {city2}" + " " * 50)
            sys.stdout.flush()
            try:
                routes[f"{city1} to {city2}"] = shortest_path_trees[city1][city2]
            except KeyError:
                print(f"No path found between {city1} and {city2}")
    return routes


# Main

start_time = time.time()

script_number = extract_script_number(__file__)

# Ensure DATADIR exists
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

# Determine input files
input_north_island_cities_path = find_data_file(INPUT_NORTH_ISLAND_CITIES_FILENAME, DATADIR)
input_south_island_cities_path = find_data_file(INPUT_SOUTH_ISLAND_CITIES_FILENAME, DATADIR)

input_north_island_roads_path = find_data_file(INPUT_NORTH_ISLAND_ROADS_FILENAME, DATADIR)
input_south_island_roads_path = find_data_file(INPUT_SOUTH_ISLAND_ROADS_FILENAME, DATADIR)


# Load processed datasets
north_island_towns = pd.read_csv(input_north_island_cities_path)
south_island_towns = pd.read_csv(input_south_island_cities_path)

with open(input_south_island_roads_path, 'rb') as f:
    south_island_roads = pickle.load(f)

with open(input_north_island_roads_path, 'rb') as f:
    north_island_roads = pickle.load(f)

# Calculate routes
print("\n    Calculate routes on South Island road network")
south_routes = calculate_routes(south_island_roads, south_island_towns)

print("\n    Calculate routes on North Island road network")
north_routes = calculate_routes(north_island_roads, north_island_towns)

# Save routes using pickle
with open(f"{DATADIR}/{script_number}_south_island_routes.pkl", "wb") as f:
    pickle.dump(south_routes, f)

with open(f"{DATADIR}/{script_number}_north_island_routes.pkl", "wb") as f:
    pickle.dump(north_routes, f)

print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")