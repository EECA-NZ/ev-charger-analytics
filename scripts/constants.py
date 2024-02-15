import pandas as pd
from shapely.geometry import Polygon

DEV_OR_PRD = 'prd'

MIN_TRANSITIONS = 30

RESAMPLE_FREQUENCY = '10T'

OUTPUTDIR = '../evroam'

PNG_DIR = f"{OUTPUTDIR}/png"

LOCALBASEURL = "http://localhost:8000"

DEPLOYMENTBASEURL = "http://aciacr.australiaeast.azurecontainer.io:8000"

DATADIR = "../data"

LOCALTIME = 'Pacific/Auckland'

POPULATION_THRESHOLD = 2000

MAX_PLAUSIBLE_PERIOD_BETWEEN_TRANSITIONS = pd.Timedelta(days=1)

IQR_MULTIPLIER = 4  # Default number of standard deviations of post-transition duration to flag as suspicious

# Optimize either "time" or "length"
WHAT_TO_OPTIMIZE = "time"

SOUTH_ISLAND_POLYGON = Polygon([
    (174.285767, -47.484937),
    (174.580971, -40.132815),
    (164.326542, -40.013923),
    (164.544060, -47.934484),
    (174.285767, -47.484937)
])

GEONAMES_USERNAME = 'evroamlocs'

# Which SQL database to get information on EVRoam chargers from
ENV_PRD = 'prd'

SMALLEST_ROADS_INCLUDED = 'tertiary'

# Define a threshold for the EV charger gap
THRESHOLD = 75  # Want to see a public EV charging station at least every 75 km

HUB_THRESHOLD = 200  # Want to see a public EV charging station at least every 200 km

PLOT_ROAD_NETWORK = False

SOUTH_ISLAND_ROAD_NETWORK_GAPS = [
    # name, (lat, lon), radius (km)
]

NORTH_ISLAND_ROAD_NETWORK_GAPS = [
    # name, (lat, lon), radius (km)
    ("otaki", (-40.755485, 175.162076), 10),
]

ADJUST_MAX_SPEED = {
    'motorway': 1.1,
    'trunk': 1.0,
    'primary': 0.8,
    'secondary': 0.7,
    'tertiary': 0.6,
    'default': 0.5
}

EV_CAR_PERCENTAGES = {
    2023: 0.01484, 2024: 0.01980, 2025: 0.02801, 2026: 0.04180, 2027: 0.06163,
    2028: 0.08915, 2029: 0.12119, 2030: 0.15614, 2031: 0.19318, 2032: 0.23240,
    2033: 0.27322, 2034: 0.31466, 2035: 0.35721, 2036: 0.40091, 2037: 0.44594,
    2038: 0.49056, 2039: 0.53385, 2040: 0.57599, 2041: 0.61633, 2042: 0.65472,
    2043: 0.69237, 2044: 0.72888, 2045: 0.76285, 2046: 0.79427, 2047: 0.82317,
    2048: 0.84958, 2049: 0.87358, 2050: 0.89502
}
