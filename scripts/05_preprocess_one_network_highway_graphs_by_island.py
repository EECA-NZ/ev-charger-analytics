import os
import pandas as pd
import networkx as nx
import osmnx as ox
from haversine import haversine
import pickle
import time
pd.options.mode.chained_assignment = None 

from helpers import extract_script_number, find_data_file
from constants import DATADIR


# Constants

INPUT_NORTH_ISLAND_CITIES_FILENAME = 'north_island_cities.csv'
INPUT_SOUTH_ISLAND_CITIES_FILENAME = 'south_island_cities.csv'
INPUT_NORTH_ISLAND_STATIONS_FILENAME = 'north_island_stations.csv'
INPUT_SOUTH_ISLAND_STATIONS_FILENAME = 'south_island_stations.csv'
INPUT_NORTH_ISLAND_HUBS_FILENAME = 'north_island_hub_sites.csv'
INPUT_SOUTH_ISLAND_HUBS_FILENAME = 'south_island_hub_sites.csv'
INPUT_NORTH_ISLAND_ROADS_FILENAME = 'north_island_highways_abley.graphml'
INPUT_SOUTH_ISLAND_ROADS_FILENAME = 'south_island_highways_abley.graphml'


# Functions

def connect_towns_to_nearest_nodes(graph, towns_df):
    valid_nodes = [node for node, degree in dict(graph.degree()).items() if degree >= 1]
    for _, town in towns_df.iterrows():
        town_point = (float(town['lat']), float(town['lng']))
        closest_node = None
        min_distance = float('inf')
        for node in valid_nodes:
            node_data = graph.nodes[node]
            node_point = (float(node_data['y']), float(node_data['x']))
            distance = haversine(town_point, node_point)
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        graph.add_edge(town['name'], closest_node)
        graph.nodes[town['name']]['x'] = town['lng']
        graph.nodes[town['name']]['y'] = town['lat']
        graph.nodes[town['name']]['is_town'] = True
    return graph


def location_dict(df, identifier='Name'):
    assert(set(df.columns) >= set([identifier, 'Locationlat', 'Locationlon']))
    loc_dict = {}
    for (name, lat, lon) in zip(df[identifier], df.Locationlat, df.Locationlon):
        loc_dict[name]=(lat, lon)
    return loc_dict


def label_graph_nodes_with_nearby_facilities(graph, facilities_dict, facility_type, radius_km=20):
    """
    Tag graph nodes that are within a certain distance from facilities.

    Parameters:
    graph (networkx.Graph): The road network graph.
    facilities_dict (dict): A dictionary with facility names as keys and tuples of (lat, lon) as values.
    facility_type (str): A string describing the type of facility (e.g., 'charger', 'hub').

    Returns:
    networkx.Graph: The graph with nodes tagged with the proximity to the specified facilities.
    """
    for facility_name, (facility_lat, facility_lon) in facilities_dict.items():
        ## Find the nearest node in the graph to this facility
        #nearest_node = ox.distance.nearest_nodes(graph, facility_lon, facility_lat)
        for node in graph.nodes:
            node_data = graph.nodes[node]
            node_point = (float(node_data['y']), float(node_data['x']))
            distance = haversine((facility_lat, facility_lon), node_point)
            if distance <= radius_km:  # defaults to 20 km radius
                # Tag the node with the facility type and name
                graph.nodes[node][f'near_{facility_type}'] = True
                graph.nodes[node][f'{facility_type}_name'] = facility_name
    return graph


def get_nearby_road_attributes(station_point, graph, radius_km=10, attribute_to_try_to_maximize='trafficVolume'):
    """
    Finds a road node near the given station_point within a specified radius and 
    returns its attributes. The function first tries to find a node within the radius 
    that maximizes the specified attribute. If no such node is found within the radius, 
    it falls back to the closest node with the attribute, regardless of the radius.

    Parameters:
    station_point (tuple): The latitude and longitude of the station.
    graph (networkx.Graph): The graph representing the road network.
    radius_km (float): The radius within which to search for a road node.
    attribute_to_try_to_maximize (str): The node attribute to maximize within the radius.

    Returns:
    dict: The attributes of the selected road node, or None if no suitable node is found.
    """

    valid_nodes = [node for node, attr in graph.nodes(data=True)
                   if graph.degree(node) >= 1 and attribute_to_try_to_maximize in attr and attr[attribute_to_try_to_maximize] is not None]

    nodes_within_radius = []
    for node in valid_nodes:
        node_data = graph.nodes[node]
        node_point = (float(node_data['y']), float(node_data['x']))
        distance = haversine(station_point, node_point)
        if distance <= radius_km:
            nodes_within_radius.append((node, graph.nodes[node][attribute_to_try_to_maximize]))

    if nodes_within_radius:
        max_attribute_node = max(nodes_within_radius, key=lambda x: x[1])[0]
        return graph.nodes[max_attribute_node]

    closest_node = None
    min_distance = float('inf')
    for node in valid_nodes:
        node_data = graph.nodes[node]
        node_point = (float(node_data['y']), float(node_data['x']))
        distance = haversine(station_point, node_point)
        if distance < min_distance:
            min_distance = distance
            closest_node = node

    return graph.nodes[closest_node] if closest_node is not None else None


def add_traffic_volumes_and_free_flow_speeds(stations_df, graph):
    for i, station in stations_df.iterrows():
        station_point = (float(station['Locationlat']), float(station['Locationlon']))
        road_attributes = get_nearby_road_attributes(station_point, graph)
        stations_df.loc[i, 'trafficVolume'] = road_attributes['trafficVolume']
        stations_df.loc[i, 'freeFlowSpeed'] = road_attributes['freeFlowSpeed']
    return stations_df

def save_graphs(graph, filename):
    with open(f'{DATADIR}/{filename}', 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)


# Main

start_time = time.time()

script_number = extract_script_number(__file__)

# Ensure DATADIR exists
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

# Identify input files
input_north_island_cities_path = find_data_file(INPUT_NORTH_ISLAND_CITIES_FILENAME, DATADIR)
input_south_island_cities_path = find_data_file(INPUT_SOUTH_ISLAND_CITIES_FILENAME, DATADIR)
input_north_island_stations_path = find_data_file(INPUT_NORTH_ISLAND_STATIONS_FILENAME, DATADIR)
input_south_island_stations_path = find_data_file(INPUT_SOUTH_ISLAND_STATIONS_FILENAME, DATADIR)
input_north_island_hubs_path = find_data_file(INPUT_NORTH_ISLAND_HUBS_FILENAME, DATADIR)
input_south_island_hubs_path = find_data_file(INPUT_SOUTH_ISLAND_HUBS_FILENAME, DATADIR)
input_north_island_roads_path = find_data_file(INPUT_NORTH_ISLAND_ROADS_FILENAME, DATADIR)
input_south_island_roads_path = find_data_file(INPUT_SOUTH_ISLAND_ROADS_FILENAME, DATADIR)

# Load datasets
north_island_towns = pd.read_csv(input_north_island_cities_path)
south_island_towns = pd.read_csv(input_south_island_cities_path)
north_island_stations = pd.read_csv(input_north_island_stations_path)
south_island_stations = pd.read_csv(input_south_island_stations_path)
north_island_hubs = pd.read_csv(input_north_island_hubs_path)
south_island_hubs = pd.read_csv(input_south_island_hubs_path)
north_island_roads = nx.read_graphml(input_north_island_roads_path)
south_island_roads = nx.read_graphml(input_south_island_roads_path)
north_island_roads.graph['crs'] = 'epsg:4326'
south_island_roads.graph['crs'] = 'epsg:4326'

print("\nPre-process South Island road network")
print("\n    Get road attributes near EV chargers")
south_island_stations = add_traffic_volumes_and_free_flow_speeds(south_island_stations, south_island_roads)
print("\n    Connect towns into road network")
south_island_roads = connect_towns_to_nearest_nodes(south_island_roads, south_island_towns)
print("\n    Add EV chargers to road network")
south_island_station_dict = location_dict(south_island_stations, identifier='Name')
south_island_roads = label_graph_nodes_with_nearby_facilities(south_island_roads, south_island_station_dict, 'charger')
print("\n    Add EV hubs to road network")
south_island_hub_dict = location_dict(south_island_hubs, identifier='Location')
south_island_roads = label_graph_nodes_with_nearby_facilities(south_island_roads, south_island_hub_dict, 'hub')

print("\nPre-process North Island road network")
print("\n    Get road attributes near EV chargers")
north_island_stations = add_traffic_volumes_and_free_flow_speeds(north_island_stations, north_island_roads)
print("\n    Connect towns into road network")
north_island_roads = connect_towns_to_nearest_nodes(north_island_roads, north_island_towns)
print("\n    Add EV chargers to road network")
north_island_station_dict = location_dict(north_island_stations, identifier='Name')
north_island_roads = label_graph_nodes_with_nearby_facilities(north_island_roads, north_island_station_dict, 'charger')
print("\n    Add EV hubs to road network")
north_island_hub_dict = location_dict(north_island_hubs, identifier='Location')
north_island_roads = label_graph_nodes_with_nearby_facilities(north_island_roads, north_island_hub_dict, 'hub')

print("Save the road network graphs")
save_graphs(south_island_roads, f"{script_number}_south_island_abley_roads_with_towns_and_ev_charging_stations.gpickle")
save_graphs(north_island_roads, f"{script_number}_north_island_abley_roads_with_towns_and_ev_charging_stations.gpickle")

print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")