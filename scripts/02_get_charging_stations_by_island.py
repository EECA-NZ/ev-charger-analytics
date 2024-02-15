import os
import pandas as pd
from ModernDataPlatform import DbInteract
from shapely.geometry import Point
from constants import DATADIR, SOUTH_ISLAND_POLYGON, ENV_PRD
import time

from helpers import extract_script_number

pd.options.mode.chained_assignment = None

# Constants

NORTH_ISLAND_CSV_OUT = 'north_island_stations.csv'
SOUTH_ISLAND_CSV_OUT = 'south_island_stations.csv'

# Functions

def get_charging_station_locations(env: str = ENV_PRD) -> pd.DataFrame:
    """
    Fetch raw data from GIDI database.
    """
    mdp_db = DbInteract(env=env)
    charging_stations = mdp_db.read("SELECT * " + \
                                 "FROM EVRoam.ChargingStations ERCS " + \
                                 "INNER JOIN EVRoam.Sites ERS ON ERCS.SiteId = ERS.SiteId " + \
                                 "WHERE ERCS.ProviderDeleted != 1 " + \
                                 "AND ERS.ProviderDeleted != 1 " + \
                                 "AND ERCS.InstallationStatus = 'Commissioned'"
                                )
    charging_stations = charging_stations.set_index('ChargingStationId')
    return charging_stations

def determine_island(row):
    point = Point(row['Locationlon'], row['Locationlat'])
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

# Fetch charging stations
charging_stations = get_charging_station_locations()

# Assuming you have 'Locationlon' and 'Locationlat' columns in the charging_stations DataFrame
charging_stations['island'] = charging_stations.apply(determine_island, axis=1)

# Save the results to separate CSV files for each island in the specified directory
charging_stations[charging_stations['island'] == 'North'].to_csv(f'{DATADIR}/{script_number}_{NORTH_ISLAND_CSV_OUT}')
charging_stations[charging_stations['island'] == 'South'].to_csv(f'{DATADIR}/{script_number}_{SOUTH_ISLAND_CSV_OUT}')

print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")