import os
import geopandas as gpd
from pyproj import Transformer
from constants import DATADIR
from shapely.ops import transform
import time

from helpers import extract_script_number, find_data_file

# Constants

GEOPACKAGE_FILENAME = 'ChargingHubSites.gpkg'


# Functions

def get_charging_hub_sites(file_path: str) -> gpd.GeoDataFrame:
    """
    Read charging hub site data from a Geopackage file, reproject to WGS84,
    and calculate centroids.
    """
    gdf = gpd.read_file(file_path)
    # Transformer to convert EPSG:2193 (NZTM) to EPSG:4326 (WGS84 lat/lon)
    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: transform(transformer.transform, geom))

    # Calculate centroid for each geometry
    gdf["Locationlat"] = gdf["geometry"].centroid.y
    gdf["Locationlon"] = gdf["geometry"].centroid.x

    return gdf

# Main

start_time = time.time()

script_number = extract_script_number(__file__)

# Ensure DATADIR exists
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

# Get file path for the geopackage
gpkg_file_path = find_data_file(GEOPACKAGE_FILENAME, DATADIR)

# Fetch, reproject, and get centroids of charging hub sites
charging_hub_sites = get_charging_hub_sites(gpkg_file_path)

# Drop the original geometry column
charging_hub_sites = charging_hub_sites.drop(columns=['geometry'])

# Filter data by island
north_island_sites = charging_hub_sites[charging_hub_sites['Island'] == 'NI']
south_island_sites = charging_hub_sites[charging_hub_sites['Island'] == 'SI']

# Save the results to separate CSV files for each island in the specified directory
north_island_sites.to_csv(f'{DATADIR}/{script_number}_north_island_hub_sites.csv', index=False)
south_island_sites.to_csv(f'{DATADIR}/{script_number}_south_island_hub_sites.csv', index=False)

print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")