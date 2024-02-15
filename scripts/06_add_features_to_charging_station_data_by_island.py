import os
import pandas as pd
import geopandas as gpd
import networkx as nx
from haversine import haversine
from shapely.geometry import Point
from constants import DATADIR
import pickle
import time
pd.options.mode.chained_assignment = None

from helpers import extract_script_number, find_data_file

# Constants
INPUT_NORTH_ISLAND_CITIES_FILENAME = 'north_island_cities.csv'
INPUT_SOUTH_ISLAND_CITIES_FILENAME = 'south_island_cities.csv'
INPUT_NORTH_ISLAND_STATIONS_FILENAME = 'north_island_stations.csv'
INPUT_SOUTH_ISLAND_STATIONS_FILENAME = 'south_island_stations.csv'
INPUT_NORTH_ISLAND_HUBS_FILENAME = 'north_island_hub_sites.csv'
INPUT_SOUTH_ISLAND_HUBS_FILENAME = 'south_island_hub_sites.csv'
INPUT_NORTH_ISLAND_ROADS_FILENAME = 'north_island_highways_abley.graphml'
INPUT_SOUTH_ISLAND_ROADS_FILENAME = 'south_island_highways_abley.graphml'
INPUT_SA2_POPN_EV_FILENAME = 'SA2_year_popn_EV.gpkg'

# Functions

def get_nearby_road_attributes(station_point,
                               graph,
                               radius_km=10, 
                               attribute_to_try_to_maximize='trafficVolume'):
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
                   if graph.degree(node) >= 1 and attribute_to_try_to_maximize in attr
                   and attr[attribute_to_try_to_maximize] is not None]

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


def get_closest_SA2_data(charger_point, gdf_SA2):
    # Calculate the distances from the charger to each SA2 region centroid
    distances = gdf_SA2_utm.geometry.centroid.distance(charger_point)
    closest_SA2_idx = distances.idxmin()
    # Return the data of the closest SA2 region
    return gdf_SA2_utm.iloc[closest_SA2_idx]


def add_SA2_data_to_chargers(gdf_chargers, gdf_SA2, columns_of_interest):
    for index, charger in gdf_chargers.iterrows():
        closest_SA2_data = get_closest_SA2_data(charger.geometry, gdf_SA2)
        for column in columns_of_interest:
            gdf_chargers.at[index, column] = closest_SA2_data[column]
    return gdf_chargers


def reproject_SA2_data():
    sa2_popn_ev = find_data_file(INPUT_SA2_POPN_EV_FILENAME, DATADIR)
    gdf_SA2 = gpd.read_file(sa2_popn_ev)
    return gdf_SA2.to_crs(epsg=32760)


def append_columns_of_interest(stations_df, gdf_SA2_utm, year):
    stations_df['geometry'] = [Point(lon, lat) for lon, lat in zip(stations_df['Locationlon'], stations_df['Locationlat'])]
    gdf_stations = gpd.GeoDataFrame(stations_df, geometry='geometry', crs="EPSG:4326")
    gdf_stations_utm = gdf_stations.to_crs(epsg=32760)
    gdf_SA2_utm_filtered = gdf_SA2_utm[gdf_SA2_utm['Year'] == year]
    gdf_SA2_utm_filtered.loc[:, 'centroid'] = gdf_SA2_utm_filtered.geometry.centroid
    gdf_SA2_centroids_filtered = gpd.GeoDataFrame(gdf_SA2_utm_filtered, geometry='centroid', columns=['centroid', *columns_of_interest])
    gdf_SA2_centroids_filtered.crs = gdf_SA2_utm.crs
    gdf_stations_with_SA2_filtered = gpd.sjoin_nearest(gdf_stations_utm, gdf_SA2_centroids_filtered, how='left', distance_col="distance")
    gdf_stations_with_SA2_filtered.drop(columns=['index_right', 'distance'], inplace=True)
    gdf_stations_with_SA2 = gdf_stations_with_SA2_filtered.to_crs(epsg=4326)
    return gdf_stations_with_SA2

def get_neighboring_stations_data(station_point, stations_df, radius_km=25):

    neighboring_stations = []
    for i, potential_neighbor in stations_df.iterrows():
        potential_neighbor_point = (float(potential_neighbor['Locationlat']), float(potential_neighbor['Locationlon']))
        distance = haversine(station_point, potential_neighbor_point)
        if 0 < distance <= radius_km:  # Exclude the station itself and check the radius
            neighboring_stations.append(potential_neighbor)
    return pd.DataFrame(neighboring_stations)

def add_neighboring_stations_data(stations_df, radius_km=20):
    # Initialize columns for neighboring stations' data
    stations_df['neighboring_station_count'] = 0
    stations_df['mean_neighboring_station_distance'] = None
    stations_df['mean_neighboring_SA2_population'] = None
    stations_df['mean_neighboring_trafficVolume'] = None
    stations_df['mean_neighboring_freeFlowSpeed'] = None

    for i, station in stations_df.iterrows():
        station_point = (float(station['Locationlat']), float(station['Locationlon']))
        neighboring_stations_df = get_neighboring_stations_data(station_point, stations_df, radius_km)
        
        # Count the number of neighboring stations
        neighbor_count = len(neighboring_stations_df)
        stations_df.at[i, 'neighboring_station_count'] = neighbor_count
        
        # Skip further processing if there are no neighboring stations
        if neighbor_count == 0:
            continue
        
        # Calculate and store mean distance of neighboring stations
        mean_distance = neighboring_stations_df.apply(
            lambda x: haversine(station_point, (x['Locationlat'], x['Locationlon'])), axis=1).mean()
        stations_df.at[i, 'mean_neighboring_station_distance'] = mean_distance

        # Check for relevant columns and calculate means if available
        if 'Population_estimate' in neighboring_stations_df.columns:
            mean_population = neighboring_stations_df['Population_estimate'].mean()
            stations_df.at[i, 'mean_neighboring_SA2_population'] = mean_population

        if 'trafficVolume' in neighboring_stations_df.columns:
            mean_traffic_volume = neighboring_stations_df['trafficVolume'].mean()
            stations_df.at[i, 'mean_neighboring_trafficVolume'] = mean_traffic_volume

        if 'freeFlowSpeed' in neighboring_stations_df.columns:
            mean_free_flow_speed = neighboring_stations_df['freeFlowSpeed'].mean()
            stations_df.at[i, 'mean_neighboring_freeFlowSpeed'] = mean_free_flow_speed

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

print("\n    Get road attributes near EV chargers - South Island")
south_island_stations = add_traffic_volumes_and_free_flow_speeds(south_island_stations, south_island_roads)

print("\n    Get road attributes near EV chargers - North Island")
north_island_stations = add_traffic_volumes_and_free_flow_speeds(north_island_stations, north_island_roads)

print("\nLoading and reprojecting SA2 data")
gdf_SA2_year_popn_EV_utm = reproject_SA2_data()
columns_of_interest = ['SA2_code', 'SA2_name', 'Year', 'Population_estimate', 'EV_1000', 'EV', 'Chargers_needed', 'Population_growth']
specific_year = 2023

print("Process stations for both North and South Islands")
gdf_north_island_chargers_with_SA2 = append_columns_of_interest(north_island_stations, gdf_SA2_year_popn_EV_utm, specific_year)
gdf_south_island_chargers_with_SA2 = append_columns_of_interest(south_island_stations, gdf_SA2_year_popn_EV_utm, specific_year)

print("Add neighbouring station data for both North and South Islands")
north_island_stations_with_neighbors = add_neighboring_stations_data(gdf_north_island_chargers_with_SA2, radius_km=25)
south_island_stations_with_neighbors = add_neighboring_stations_data(gdf_south_island_chargers_with_SA2, radius_km=25)

# Save the final dataframe that contains all the columns
north_island_stations_with_neighbors.to_csv(f'{DATADIR}/{script_number}_north_island_stations_complete.csv', index=False)
south_island_stations_with_neighbors.to_csv(f'{DATADIR}/{script_number}_south_island_stations_complete.csv', index=False)


print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")