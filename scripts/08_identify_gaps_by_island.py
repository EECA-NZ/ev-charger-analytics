import os
import time
import pickle
import pandas as pd
from haversine import haversine

from helpers import extract_script_number, find_data_file

#### constants

from constants import DATADIR, THRESHOLD, HUB_THRESHOLD


INPUT_NORTH_ISLAND_CITIES_FILENAME = 'north_island_cities.csv'
INPUT_SOUTH_ISLAND_CITIES_FILENAME = 'south_island_cities.csv'

INPUT_NORTH_ISLAND_ROADS_FILENAME = 'north_island_abley_roads_with_towns_and_ev_charging_stations.gpickle'
INPUT_SOUTH_ISLAND_ROADS_FILENAME = 'south_island_abley_roads_with_towns_and_ev_charging_stations.gpickle'

INPUT_NORTH_ISLAND_ROUTES_FILENAME = 'north_island_routes.pkl'
INPUT_SOUTH_ISLAND_ROUTES_FILENAME = 'south_island_routes.pkl'


#### functions

def compute_distances_from_last_facility(graph, route, facility_type):
    distance_since_last_facility = 0.0
    distances = []
    last_facility_name = route[0]
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        coord_u = (float(graph.nodes[u]['y']), float(graph.nodes[u]['x']))
        coord_v = (float(graph.nodes[v]['y']), float(graph.nodes[v]['x']))
        distance = haversine(coord_u, coord_v)
        distance_since_last_facility += distance
        # Check if node is near a facility of type {facility_type}
        if graph.nodes[u].get(f'near_{facility_type}', False):
            last_facility_name = graph.nodes[u].get(f'{facility_type}_name', last_facility_name)
            distance_since_last_facility = 0.0
        elif graph.nodes[v].get(f'near_{facility_type}', False):
            last_facility_name = graph.nodes[v].get(f'{facility_type}_name', last_facility_name)
            distance_since_last_facility = 0.0
        distances.append((distance_since_last_facility, last_facility_name))
    return distances

def get_population(city, island_towns):
    row = island_towns[island_towns["name"] == city]
    if not row.empty:
        return row.iloc[0]["population"]
    return 0

def prioritize_gaps(gaps, island_towns):
    for node, reasons in gaps.items():
        reasons.sort(key=lambda x: (-x["distance"],
                                    -get_population(x["route_name"].split(" to ")[0], island_towns),
                                    -get_population(x["route_name"].split(" to ")[1], island_towns)))
        # Retain only the top reason
        gaps[node] = reasons[0]
    return gaps

def identify_gaps_in_routes(roads, routes, towns, facility_type, threshold):
    routes_distances = {}
    identified_gaps = {}
    # Iterate over all route names
    for route_key in routes:
        if not routes[route_key]:  # Skip None or empty routes
            print(f"Skipping {route_key} as it's None or empty")
            continue
        city1, city2 = route_key.split(" to ")
        # Compute distances from last charger
        routes_distances[route_key] = compute_distances_from_last_facility(roads, routes[route_key], facility_type)
        # Identify gaps greater than the threshold
        for node, (distance, charger_name) in zip(routes[route_key], routes_distances[route_key]):
            if distance > threshold:
                if node not in identified_gaps:
                    identified_gaps[node] = []
                identified_gaps[node].append({
                    "distance": distance,
                    f"{facility_type}_name": charger_name,
                    "route_name": route_key
                })
    identified_gaps = prioritize_gaps(identified_gaps, towns)
    routes_with_gaps = set()
    for node in identified_gaps:
        route_name = identified_gaps[node]["route_name"]
        routes_with_gaps.add(route_name)
    return routes_distances, identified_gaps, routes_with_gaps



#### main

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

input_north_island_routes_path = find_data_file(INPUT_NORTH_ISLAND_ROUTES_FILENAME, DATADIR)
input_south_island_routes_path = find_data_file(INPUT_SOUTH_ISLAND_ROUTES_FILENAME, DATADIR)


with open(input_south_island_roads_path, 'rb') as f:
    south_island_roads = pickle.load(f)

with open(input_north_island_roads_path, 'rb') as f:
    north_island_roads = pickle.load(f)

with open(input_south_island_routes_path, "rb") as f:
    south_routes = pickle.load(f)

with open(input_north_island_routes_path, "rb") as f:
    north_routes = pickle.load(f)

north_island_towns = pd.read_csv(input_north_island_cities_path)
south_island_towns = pd.read_csv(input_south_island_cities_path)

print("Identify gaps in EV charger network in North Island")
_, north_identified_charger_gaps, north_island_routes_with_charger_gaps = identify_gaps_in_routes(north_island_roads, north_routes, north_island_towns, "charger", THRESHOLD)

print("Identify gaps in EV charger network in South Island")
_, south_identified_charger_gaps, south_island_routes_with_charger_gaps = identify_gaps_in_routes(south_island_roads, south_routes, south_island_towns, "charger", THRESHOLD)

print("Identify gaps in EV charger hub network in North Island")
_, north_identified_hub_gaps, north_island_routes_with_hub_gaps = identify_gaps_in_routes(north_island_roads, north_routes, north_island_towns, "hub", HUB_THRESHOLD)

print("Identify gaps in EV charger hub network in South Island")
_, south_identified_hub_gaps, south_island_routes_with_hub_gaps = identify_gaps_in_routes(south_island_roads, south_routes, north_island_towns, "hub", HUB_THRESHOLD)

print("Consolidate identified gaps")
north_island_gaps = {
    'hub_gaps': north_identified_hub_gaps,
    'hub_gap_routes': north_island_routes_with_hub_gaps,
    'charger_gaps': north_identified_charger_gaps,
    'charger_gap_routes': north_island_routes_with_charger_gaps
}

south_island_gaps = {
    'hub_gaps': south_identified_hub_gaps,
    'hub_gap_routes': south_island_routes_with_hub_gaps,
    'charger_gaps': south_identified_charger_gaps,
    'charger_gap_routes': south_island_routes_with_charger_gaps
}

with open(f"{DATADIR}/{script_number}_north_island_gaps.pkl", "wb") as f:
    pickle.dump(north_island_gaps, f)

with open(f"{DATADIR}/{script_number}_south_island_gaps.pkl", "wb") as f:
    pickle.dump(south_island_gaps, f)

print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")