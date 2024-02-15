import re
import os
import time
import numpy as np
import networkx as nx
import geopandas as gpd
from pyproj import Transformer
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from shapely.ops import transform, linemerge
from shapely.geometry import MultiLineString, LineString, Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from haversine import haversine
from shapely.geometry import Point
from constants import DATADIR, SOUTH_ISLAND_POLYGON

import networkx as nx
from scipy.spatial import cKDTree
import numpy as np

from helpers import report_runtime, extract_script_number, find_data_file

# Constants
PRECISION = 3 # number of decimal places to round coordinates to

INPUT_GPKG_FILENAME = 'StateHighway_view.gpkg'
SOUTH_ISLAND_GRAPHML_FILENAME = "south_island_highways_abley.graphml"
NORTH_ISLAND_GRAPHML_FILENAME = "north_island_highways_abley.graphml"

SMALL_RADIUS_KM = 1 # to join separate subgraphs, connect vertices within this radius
LARGE_RADIUS_KM = 2 # to check whether joined, look at subgraph within this radius
LENGTH_MULTIPLIER = np.sqrt(2) # when joining separate subgraphs, multiply haversine distance by this factor to estimate path length
PRECISION = 5 # number of decimal places to round coordinates to

# Functions

def determine_island(geometry):
    # Extract the first point from the geometry
    if isinstance(geometry, MultiLineString):
        # Access the first LineString in the MultiLineString
        first_linestring = geometry.geoms[0]  # Access the first LineString
        first_point = first_linestring.coords[0]
    elif isinstance(geometry, LineString):
        first_point = geometry.coords[0]
    else:
        raise TypeError("Unsupported geometry type")
    point = Point(first_point[0], first_point[1])
    if SOUTH_ISLAND_POLYGON.contains(point):
        return "South"
    else:
        return "North"


def load_gdf(gpkg_path):
    # Load the GeoDataFrame
    gdf = gpd.read_file(gpkg_path)
    # Transformer to convert EPSG:2193 (NZTM) to EPSG:4326 (WGS84 lat/lon)
    transformer = Transformer.from_crs('EPSG:2193', 'EPSG:4326', always_xy=True)
    # Convert the geometry to WGS84 lat/lon
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: transform(transformer.transform, geom))
    # Label the island of each row
    gdf['island'] = gdf['geometry'].apply(determine_island)
    return gdf


def round_coordinates(coord, precision):
    """Round coordinates to a specific precision."""
    return tuple(round(c, precision) for c in coord)


def contract_graph(G, max_edge_length_km=1.0):
    """
    Simplify the graph by contracting nodes with degree 2, up to a maximum edge length.
    This function assumes that the graph G has an edge attribute 'length'.
    """
    simplified_G = G.copy()
    nodes_to_contract = [node for node in G if simplified_G.degree(node) == 2]
    while nodes_to_contract:
        node = nodes_to_contract.pop()
        if simplified_G.degree(node) != 2:
            continue
        neighbors = list(simplified_G.neighbors(node))
        if len(neighbors) != 2:
            continue
        # Calculate the total length of edges to be combined
        total_length = simplified_G[node][neighbors[0]]['length'] + simplified_G[node][neighbors[1]]['length']
        total_time = simplified_G[node][neighbors[0]]['time'] + simplified_G[node][neighbors[1]]['time']
        # Only contract if the total length is within the threshold
        if total_length <= max_edge_length_km:
            # Add a new edge with the combined length
            simplified_G.add_edge(neighbors[0], neighbors[1], length=total_length, time=total_time)
            # Remove the contracted node
            simplified_G.remove_node(node)
    return simplified_G


def closest_nodes_in_distinct_subgraphs(subgraphs, G, max_degree=float('inf')):
    min_distance = float('inf')
    closest_pair = None
    for i, sg1 in enumerate(subgraphs):
        for j, sg2 in enumerate(subgraphs):
            if i >= j:  # Skip same subgraph comparison
                continue
            # Find periphery nodes for each subgraph
            sg1_periphery = [n for n in sg1 if G.degree(n) <= max_degree]
            sg2_periphery = [n for n in sg2 if G.degree(n) <= max_degree]
            # Find the closest pair of nodes between sg1 and sg2
            for node1 in sg1_periphery:
                for node2 in sg2_periphery:
                    distance = haversine(node1, node2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (node1, node2)
    return closest_pair


def patch_road_network_iteratively(G, max_degree=float('inf')):
    iteration = 0
    while True:
        iteration += 1
        # Get the list of all subgraphs
        subgraphs = list(nx.connected_components(G))
        if len(subgraphs) == 1:
            print("All subgraphs are connected. Stopping.")
            break
        print(f"Iteration {iteration}. Number of disconnected subgraphs: {len(subgraphs)}...")
        # Find the closest subgraphs and connect them
        node1, node2 = closest_nodes_in_distinct_subgraphs(subgraphs, G, max_degree=max_degree)
        distance = haversine(node1, node2)
        G.add_edge(node1, node2, length=distance, time=distance / 50)
    print(f"Completed patching after {iteration} iterations.")
    return G


def connect_nearby_vertices(G, small_radius_km=SMALL_RADIUS_KM, large_radius_km=LARGE_RADIUS_KM, length_multiplier=LENGTH_MULTIPLIER):
    # Extract node coordinates and create a mapping from KDTree indices to graph nodes
    node_coords = np.array([(G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes])
    idx_to_node = list(G.nodes)
    tree = cKDTree(node_coords)

    # Iterate over each node to find and connect nearby vertices
    for idx, coords_i in enumerate(node_coords):
        i = idx_to_node[idx]
        # Find all points within the small radius
        points_within_small_radius = tree.query_ball_point(coords_i, small_radius_km / 111)  # convert km to degrees

        # Extract the subgraph within the large radius
        points_within_large_radius = tree.query_ball_point(coords_i, large_radius_km / 111)  # convert km to degrees
        sg_nodes = [idx_to_node[idx] for idx in points_within_large_radius]
        sg = G.subgraph(sg_nodes)

        # Connect vertices that are not already connected in the subgraph
        for j_idx in points_within_small_radius:
            if j_idx != idx:
                j = idx_to_node[j_idx]
                if not sg.has_edge(i, j):
                    coords_j = node_coords[j_idx]
                    distance_km = haversine(coords_i, coords_j)
                    weighted_distance = distance_km * length_multiplier
                    G.add_edge(i, j, length=weighted_distance, time=weighted_distance / 100)

    return G


def save_roads(G, file_name):
    nx.write_graphml(G, f"{DATADIR}/{file_name}")
    print(f"Saved {file_name} to {DATADIR}")


def is_valid_coordinate_string(s):
    """Check if the string s is a valid representation of a coordinate pair."""
    pattern = r'^\(\-?\d+(\.\d+)?, \-?\d+(\.\d+)?\)$'
    return re.match(pattern, s) is not None


def convert_node_identifiers_to_latlon(G):
    """Convert string node identifiers to tuple coordinates, with a regex check."""
    mapping = {}
    for node in G.nodes():
        if is_valid_coordinate_string(node):
            mapping[node] = eval(node)
        else:
            raise ValueError(f"Invalid node identifier format: {node}")
    return nx.relabel_nodes(G, mapping)


def graph_from_gdf(gdf, precision=PRECISION):
    nodes, edges = {}, {}
    for index, row in gdf.iterrows():
        attributes = {
            'streetCategory': row['streetCategory'],
            'roadStereotype': row['roadStereotype'],
            'speedLimit': row['speedLimit'],
            'trafficVolume': row['trafficVolume'],
            'aadtBand': row['aadtBand'],
            'width': row['width'],
            'freeFlowSpeed': row['freeFlowSpeed'],
            'island': row['island']
        }
        geometry = row['geometry']
        previous_node = None
        for geom in row.geometry.geoms:
            for node in geom.coords:
                rounded_node = round_coordinates(node, precision)[::-1]
                nodes[rounded_node] = attributes.copy()
                nodes[rounded_node]['x'] = rounded_node[1]
                nodes[rounded_node]['y'] = rounded_node[0]
                if previous_node is not None:
                    edge = (previous_node, rounded_node)
                    edges[edge] = attributes.copy()
                    edges[edge]['length'] = haversine(previous_node, rounded_node)
                    edges[edge]['time'] = edges[edge]['length'] / attributes['freeFlowSpeed']
                previous_node = rounded_node
    G = nx.Graph()
    for n in nodes:
        G.add_node(n, **nodes[n])
    for e in edges:
        G.add_edge(e[0], e[1], **edges[e])
    return G


def build_kdtree(subgraph, G):
    # Extract x, y coordinates for nodes in the subgraph
    coords = [(G.nodes[node]['x'], G.nodes[node]['y']) for node in subgraph]
    # Build and return a KDTree for these coordinates
    return cKDTree(coords), coords

def find_closest_nodes_between_subgraphs(subgraph1, subgraph2, G):
    tree1, coords1 = build_kdtree(subgraph1, G)
    tree2, coords2 = build_kdtree(subgraph2, G)
    
    # Find the closest points between the two sets of coordinates
    dist, idx = tree1.query(coords2, k=1)
    closest_pair_idx = np.argmin(dist)
    closest_dist = dist[closest_pair_idx]
    
    # Retrieve the actual nodes corresponding to these indices
    node1 = list(subgraph1)[idx[closest_pair_idx]]
    node2 = list(subgraph2)[closest_pair_idx]
    
    return node1, node2, closest_dist

def patch_road_network_iteratively(G, max_degree=float('inf')):
    iteration = 0

    while True:
        iteration += 1
        subgraphs = list(nx.connected_components(G))
        if len(subgraphs) == 1:
            print("All subgraphs are connected. Stopping.")
            break
        print(f"Iteration {iteration}. Number of disconnected subgraphs: {len(subgraphs)}...")
        
        # Initialize a large distance to compare against
        min_distance = float('inf')
        for i, subgraph1 in enumerate(subgraphs[:-1]):
            for subgraph2 in subgraphs[i+1:]:
                node1, node2, distance = find_closest_nodes_between_subgraphs(subgraph1, subgraph2, G)
                if distance < min_distance:
                    min_distance = distance
                    closest_node1, closest_node2 = node1, node2
        
        # Connect the closest nodes found
        if min_distance < float('inf'):
            G.add_edge(closest_node1, closest_node2, length=min_distance)
        
        print(f"Connected nodes {closest_node1} and {closest_node2} with distance {min_distance}")
    
    print(f"Completed patching after {iteration} iterations.")
    return G


def create_graph(graphml_path, gdf_path, precision=PRECISION, island='South'):
    """Create the graph from a GeoDataFrame."""
    print(f"Load {island} island data from GeoDataFrame...")
    gdf = load_gdf(gdf_path)
    island_gdf = gdf[gdf['island'] == island]
    print("Create graph from GeoDataFrame...")
    G = graph_from_gdf(island_gdf, precision=precision)
    #G = contract_graph(G)
    print("Connect nearby vertices...")
    G = connect_nearby_vertices(G, small_radius_km=SMALL_RADIUS_KM, large_radius_km=LARGE_RADIUS_KM, length_multiplier=LENGTH_MULTIPLIER)
    print("Patch road network iteratively...")
    G = patch_road_network_iteratively(G)
    nx.write_graphml(G, graphml_path)
    return G


start_time = time.time()

script_number = "04"

input_gpkg_path = find_data_file(INPUT_GPKG_FILENAME, DATADIR)

south_island_output_path = f"{DATADIR}/{script_number}_{SOUTH_ISLAND_GRAPHML_FILENAME}"
north_island_output_path = f"{DATADIR}/{script_number}_{NORTH_ISLAND_GRAPHML_FILENAME}"

# Create and patch simplified graphs for each island
GSI = create_graph(south_island_output_path, input_gpkg_path, precision=PRECISION, island="South")
GNI = create_graph(north_island_output_path, input_gpkg_path, precision=PRECISION, island="North")

report_runtime(start_time)
