"""
Combined maps: make_evroam_map.py + 6_plot_road_network_and_example_routes.py

To run this script, we need to run the following scripts first:

1. 1_get_nz_towns_by_island.py
2. 2_get_charging_stations_by_island.py
3. 3_fetch_highway_network_by_island.py
4. 4_preprocess_road_networks_by_island.py
5. 5_optimal_routes_between_towns_by_island.py
6. 6_get_charging_station_availabilities.py

"""
import os
import time
import pickle
import folium
import argparse
import pandas as pd
from pathlib import Path
from folium import CustomIcon
from folium import Marker, CircleMarker
from folium import PolyLine
from folium.plugins import FloatImage
from branca.colormap import LinearColormap
from haversine import haversine

from helpers import generate_html_content, extract_script_number, find_data_file


# Constants

from constants import OUTPUTDIR, DATADIR, LOCALBASEURL, PLOT_ROAD_NETWORK

INPUT_NORTH_ISLAND_CITIES_FILENAME = 'north_island_cities.csv'
INPUT_SOUTH_ISLAND_CITIES_FILENAME = 'south_island_cities.csv'
INPUT_NORTH_ISLAND_ROADS_FILENAME = 'north_island_abley_roads_with_towns_and_ev_charging_stations.gpickle'
INPUT_SOUTH_ISLAND_ROADS_FILENAME = 'south_island_abley_roads_with_towns_and_ev_charging_stations.gpickle'
INPUT_NORTH_ISLAND_ROUTES_FILENAME = 'north_island_routes.pkl'
INPUT_SOUTH_ISLAND_ROUTES_FILENAME = 'south_island_routes.pkl'
INPUT_NORTH_ISLAND_STATIONS_FILENAME = 'north_island_stations_with_volumes_and_SA2.csv'
INPUT_SOUTH_ISLAND_STATIONS_FILENAME = 'south_island_stations_with_volumes_and_SA2.csv'
INPUT_CHARGING_STATION_TRANSITIONS_FILENAME = 'charging_station_transitions.gpickle'
INPUT_NORTH_ISLAND_HUBS_FILENAME = 'north_island_hub_sites.csv'
INPUT_SOUTH_ISLAND_HUBS_FILENAME = 'south_island_hub_sites.csv'
INPUT_NORTH_ISLAND_GAPS_FILENAME = 'north_island_gaps.pkl'
INPUT_SOUTH_ISLAND_GAPS_FILENAME = 'south_island_gaps.pkl'


# Functions

def plot_road_network(graph, map_object, color='grey', town_edge_color='magenta'):
    """# Plot the road networks"""
    for u, v, data in graph.edges(data=True):
        point_list = [[float(graph.nodes[u]['y']), float(graph.nodes[u]['x'])],
                      [float(graph.nodes[v]['y']), float(graph.nodes[v]['x'])]]
        edge_color = town_edge_color if graph.nodes[u].get('is_town', False) or graph.nodes[v].get('is_town', False) else color
        PolyLine(locations=point_list, color=edge_color, weight=2.5, opacity=1).add_to(map_object)

def plot_route_on_map(graph, m, route, color):
    """Plot a route on the map."""
    coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
    PolyLine(locations=coords, color=color, weight=2.5, opacity=1, dash_array='5, 5').add_to(m)

def rgba_to_css(rgba):
    """Convert RGBA tuple to a CSS-friendly string."""
    return f"rgba({rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]})"

def visualize_identified_gaps_on_map(graph, m, identified_gaps, kind="charger"):
    for node, gap_data in identified_gaps.items():
        #if not is_node_on_main_road(graph.nodes[node]):  # Skip if not on main road
        #    continue
        fill_color = 'yellow'
        color = rgba_to_css((255, 0, 0, 0.5))
        fill_opacity = 0.5
        if kind == "hub":
            color = '#fc7303'
            fill_color = '#ffc599'
            fill_opacity = 1
        distance = gap_data["distance"]
        name = gap_data[f"{kind}_name"]
        route_name = gap_data["route_name"]
        radius = 5 #radius = 5 + (distance - THRESHOLD) / 10  # dynamic radius based on gap size
        popup_text = f"{route_name}: {distance:.2f} km since {kind} {name}"
        CircleMarker(
            location=(graph.nodes[node]['y'], graph.nodes[node]['x']),
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            popup=popup_text
        ).add_to(m)

def plot_exists(station_id, plot_type):
    """Checks if a specific plot exists for a station."""
    return os.path.exists(f"../evroam/png/{plot_type}_{station_id}.png")

def format_tooltip(name, stats):
    tooltip_content = f'<b>{name}:</b><br>'
    for key, value in stats.items():
        if key != 'Average Charge Event Duration HH:MM:SS' or value != 'N/A':  # Skip if 'N/A'
            tooltip_content += f"{key}: {value}<br>"
    return tooltip_content

def calculate_circle_color(events_missed):
    colormap = LinearColormap(colors=['green', 'red'], index=[0, 2], vmin=0, vmax=2)
    return colormap(events_missed)

def calculate_circle_size(events):
    # You can adjust the base size and scaling factor as needed
    base_size = 5  # The base size for a circle
    scaling_factor = 2  # How much the size should increase per event
    return (base_size**2 + 5*scaling_factor*events)**0.5

def plot_evroam_map(charging_stations, site_stats, base_url="http://localhost:8000"):
    m = folium.Map(location=[-40.9006, 172.8860], zoom_start=6)

    for idx, row in charging_stations.iterrows():
        location = (row['Locationlat'], row['Locationlon'])

        custom_icon = CustomIcon(
            icon_image="../evroam/au2530-1.png",
            icon_size=(6, 6)
        )

        # Use URL parameters to pass the station details
        if plot_exists(row.name, 'availability_pattern'):
            iframe = folium.IFrame(
                f'<iframe src="{base_url}/availability_pattern.html?ChargingStationId={row.name}&StationName={row["Name"]}" width="500" height="340"></iframe>',
                width=540,
                height=360
            )
            popup = folium.Popup(iframe, max_width=520)
            circle_color = 'blue'
            circle_opacity = 1.0
            circle_size = 5
            create_marker = True

        else:
            popup_content = "Usage details unavailable for this station."
            popup = folium.Popup(popup_content, max_width=500)
            circle_color = 'grey'  # to signify lack of data
            circle_opacity = 0.5
            circle_size = 5
            create_marker = False

        if create_marker:
            stats = site_stats[row.name]
            tooltip_info = format_tooltip(row['Name'], stats)
            circle_size = calculate_circle_size(stats['Average Daily Charges'])
            circle_color = calculate_circle_color(stats['Missed Daily Charges'])
            folium.Marker(
                location=location,
                popup=popup,
                tooltip=tooltip_info,
                icon=custom_icon
            ).add_to(m)

        CircleMarker(
            location=location,
            radius=circle_size,
            color=circle_color,
            fill=False,
            opacity=circle_opacity
        ).add_to(m)

    # Relative path to your legend image
    legend_image_path = "../evroam/Legend.png"

    # Create a FloatImage object (adjust bottom and left for positioning)
    float_image = FloatImage(legend_image_path, top=10, left=5)

    # Add it to the map
    float_image.add_to(m)

    return m

def plot_charging_hubs(hubs_df, map_object):
    for _, hub in hubs_df.iterrows():
        location = (hub['Locationlat'], hub['Locationlon'])
        tooltip_text = f"{hub['Location']} EV Fast Charging Hub"  
        tooltip = folium.Tooltip(tooltip_text)  
        html = '''
        <div style="
            width: 20px;
            height: 20px;
            background-color: #0F52BA;
            color: white;
            font-size: 14px;
            font-weight: bold;
            line-height: 20px;
            text-align: center;
            border-radius: 50%;
        ">H</div>
        '''
        icon = folium.DivIcon(html=html)
        folium.Marker(location=location, icon=icon, tooltip=tooltip).add_to(map_object)




#### Main

start_time = time.time()

script_number = extract_script_number(__file__)

# Ensure DATADIR exists
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)


# Determine input files

input_north_island_cities_path = find_data_file(INPUT_NORTH_ISLAND_CITIES_FILENAME, DATADIR)
input_south_island_cities_path = find_data_file(INPUT_SOUTH_ISLAND_CITIES_FILENAME, DATADIR)
#input_north_island_stations_path = find_data_file(INPUT_NORTH_ISLAND_STATIONS_FILENAME, DATADIR)
#input_south_island_stations_path = find_data_file(INPUT_SOUTH_ISLAND_STATIONS_FILENAME, DATADIR)
input_north_island_roads_path = find_data_file(INPUT_NORTH_ISLAND_ROADS_FILENAME, DATADIR)
input_south_island_roads_path = find_data_file(INPUT_SOUTH_ISLAND_ROADS_FILENAME, DATADIR)
input_north_island_routes_path = find_data_file(INPUT_NORTH_ISLAND_ROUTES_FILENAME, DATADIR)
input_south_island_routes_path = find_data_file(INPUT_SOUTH_ISLAND_ROUTES_FILENAME, DATADIR)
input_north_island_stations_path = find_data_file(INPUT_NORTH_ISLAND_STATIONS_FILENAME, DATADIR)
input_south_island_stations_path = find_data_file(INPUT_SOUTH_ISLAND_STATIONS_FILENAME, DATADIR)
input_charging_station_transitions_path = find_data_file(INPUT_CHARGING_STATION_TRANSITIONS_FILENAME, DATADIR)
input_north_island_hubs_path = find_data_file(INPUT_NORTH_ISLAND_HUBS_FILENAME, DATADIR)
input_south_island_hubs_path = find_data_file(INPUT_SOUTH_ISLAND_HUBS_FILENAME, DATADIR)
input_north_island_gaps_path = find_data_file(INPUT_NORTH_ISLAND_GAPS_FILENAME, DATADIR)
input_south_island_gaps_path = find_data_file(INPUT_SOUTH_ISLAND_GAPS_FILENAME, DATADIR)


parser = argparse.ArgumentParser(description="Generate EV Roam map and associated content.")
parser.add_argument('--base_url', type=str, default=LOCALBASEURL, help="Base URL for the application (default: http://localhost:8000).")
args = parser.parse_args()
base_url = args.base_url


# Load processed datasets
north_island_towns = pd.read_csv(input_north_island_cities_path)
south_island_towns = pd.read_csv(input_south_island_cities_path)
north_island_ev_chargers = pd.read_csv(input_north_island_stations_path)
south_island_ev_chargers = pd.read_csv(input_south_island_stations_path)

with open(input_south_island_roads_path, 'rb') as f:
    south_island_roads = pickle.load(f)

with open(input_north_island_roads_path, 'rb') as f:
    north_island_roads = pickle.load(f)

with open(input_south_island_routes_path, "rb") as f:
    south_routes = pickle.load(f)

with open(input_north_island_routes_path, "rb") as f:
    north_routes = pickle.load(f)

charging_station_locations = pd.concat(
    [pd.read_csv(input_north_island_stations_path, index_col='SiteId'),
     pd.read_csv(input_south_island_stations_path, index_col='SiteId')]
    )

with open(input_charging_station_transitions_path, 'rb') as f:
    (status, resampled, plugin_events, queued_events, hourly_percentages, implied_demand, site_stats, all_hourly_rates) = pickle.load(f)

# Load charging hub data
north_island_charging_hubs = pd.read_csv(input_north_island_hubs_path)
south_island_charging_hubs = pd.read_csv(input_south_island_hubs_path)



print("Create map")

with open(input_north_island_gaps_path, "rb") as f:
    north_island_gaps = pickle.load(f)

with open(input_south_island_gaps_path, "rb") as f:
    south_island_gaps = pickle.load(f)


outputdir = Path(OUTPUTDIR)
outputdir.parent.mkdir(parents=True, exist_ok=True)

m = plot_evroam_map(charging_station_locations, site_stats, base_url=base_url)

print("Plot towns and EV chargers")
for _, town in north_island_towns.iterrows():
    town_icon = CustomIcon(icon_image="../static/icons/1152479-200.png", icon_size=(15, 15))
    Marker(location=[town["lat"], town["lng"]], icon=town_icon).add_to(m)

for _, town in south_island_towns.iterrows():
    town_icon = CustomIcon(icon_image="../static/icons/1152479-200.png", icon_size=(15, 15))
    Marker(location=[town["lat"], town["lng"]], icon=town_icon).add_to(m)

if PLOT_ROAD_NETWORK:
    plot_road_network(north_island_roads, m)
    plot_road_network(south_island_roads, m)

for route_name in north_island_gaps['charger_gap_routes']:
    plot_route_on_map(north_island_roads, m, north_routes[route_name], color='purple')

for route_name in south_island_gaps['charger_gap_routes']:
    plot_route_on_map(south_island_roads, m, south_routes[route_name], color='purple')

for route_name in north_island_gaps['hub_gap_routes']:
    plot_route_on_map(north_island_roads, m, north_routes[route_name], color='#fc7303')  

for route_name in south_island_gaps['hub_gap_routes']:
    plot_route_on_map(south_island_roads, m, south_routes[route_name], color='#fc7303')  

print("Visualize charger gaps")
visualize_identified_gaps_on_map(south_island_roads, m, south_island_gaps['charger_gaps'], kind="charger")
visualize_identified_gaps_on_map(north_island_roads, m, north_island_gaps['charger_gaps'], kind="charger")

print("Visualize hub gaps")
visualize_identified_gaps_on_map(south_island_roads, m, south_island_gaps['hub_gaps'], kind="hub")
visualize_identified_gaps_on_map(north_island_roads, m, north_island_gaps['hub_gaps'], kind="hub")

print("Plot charging hubs for North and South Islands")
plot_charging_hubs(north_island_charging_hubs, m)
plot_charging_hubs(south_island_charging_hubs, m)

m.save(f"{OUTPUTDIR}/index.html")

html_content = generate_html_content()
with open(f"{OUTPUTDIR}/availability_pattern.html", "w") as file:
    file.write(html_content)

print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")