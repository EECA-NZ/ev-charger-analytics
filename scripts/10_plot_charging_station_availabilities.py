import os
import sys
import glob
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

from helpers import freq_to_hours, extract_script_number, find_data_file


# Constants
from constants import MIN_TRANSITIONS, OUTPUTDIR, DATADIR, RESAMPLE_FREQUENCY, PNG_DIR

FIGSIZE = (15, 2)
AVAILABILITY_STATUS = ['Available', 'Unavailable', 'Occupied']
STATUS_MAP = {'Available': 0.0, 'Unavailable': 0.5, 'Occupied': 1.0}

INPUT_NORTH_ISLAND_STATIONS_FILENAME = 'north_island_stations_with_volumes_and_SA2.csv'
INPUT_SOUTH_ISLAND_STATIONS_FILENAME = 'south_island_stations_with_volumes_and_SA2.csv'
INPUT_CHARGING_STATION_TRANSITIONS_FILENAME = 'charging_station_transitions.gpickle'


# Functions

def plot_evroam_availability_pattern(hourly_percentages, resample_period, **kwargs):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    try:
        plt.tight_layout(pad=5)

        # Define previous layer height for stacked filling
        prev_layer_height = hourly_percentages[[]].copy()  # start with an empty dataframe having the same index
        prev_layer_height["height"] = 0

        colors = {'Occupied': 'red', 'Unavailable': 'yellow', 'Available': 'green'}
        layers = {}
        for status in AVAILABILITY_STATUS:
            if status in hourly_percentages.columns:
                layers[status] = (prev_layer_height["height"].copy(), (prev_layer_height["height"] + hourly_percentages[status]).copy())
                prev_layer_height["height"] += hourly_percentages[status]  # update the previous layer height           

        # Plot the layers in desired order
        for status in AVAILABILITY_STATUS[::-1]:
            if status in layers:
                bottom, top = layers[status]
                ax.fill_between(hourly_percentages.index, bottom, top, color=colors[status], alpha=0.5, label=status)
                ax.plot(hourly_percentages.index, top, color=colors[status], linewidth=2, alpha=0.8)

        title = "Daily Usage Status Percentages"
        if 'charger_name' in kwargs:
            charger_name = kwargs['charger_name']
            title += f" at {charger_name}"
        ax.set_title(title)

        ax.set_position([0.1, 0.1, 0.85, 0.85])  # [left, bottom, width, height]
        ax.set_xlabel(f"New Zealand local time of day grouped by {int(resample_period*60)} minutes")
        ax.set_ylim(0, 100)  # set y-axis limits to 0 to 100

       # Adjust the xlim based on the time grouping key
        ax.set_xlim(hourly_percentages.index.min(), hourly_percentages.index.max())

        # Set the x-ticks to represent hours and label them
        ticks = np.linspace(hourly_percentages.index.min(), hourly_percentages.index.max(), num=25)  # 25 because 0-24 inclusive
        #ticks = np.linspace(grouped.index.min(), grouped.index.max(), num=25)  # 25 because 0-24 inclusive
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(tick)) for tick in np.arange(0, 25, 1)])  # Adjusted to include 24

        ax.legend(title="Usage Status", loc='upper left')

        # Save the figure if output_path is provided
        if 'output_path' in kwargs:
            output_path = Path(kwargs['output_path'])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()

    finally:
        plt.close(fig)


def plot_evroam_availabilities(transitions, plugin_events, suspicious_mask=None, **kwargs):
    """
    Plot the availability statuses of an EVRoam charger, highlighting suspicious data points.
    Parameters:
    - transitions: DataFrame containing charger availability data.
    - suspicious_mask: Optional boolean Series to indicate suspicious data points.
    - kwargs: Additional keyword arguments like 'charger_name' or 'output_path'.
    """
    # Create a new figure
    fig, ax = plt.subplots(figsize=FIGSIZE)

    try:

        # Status encoding
        transitions['status_val'] = transitions['AvailabilityStatus'].map(STATUS_MAP)

        # Plotting the data
        ax.step(transitions['LocalTimePeriod'], transitions['status_val'], where='post', color='k')

        # Plotting the suspicious data in red, if mask is provided
        if suspicious_mask is not None:
            # Ensure the mask aligns with the transitions index
            suspicious_mask = suspicious_mask.reindex(transitions.index, fill_value=False)
            for i in range(len(transitions) - 1):
                if suspicious_mask.iloc[i]:
                    transition_segment = transitions.iloc[i:i+2]
                    ax.step(transition_segment['LocalTimePeriod'].iloc[::-1], transition_segment['status_val'], where='post', color='r')

        # Marking plugin events
        plugin_times = plugin_events['LocalTimePeriod']
        plugin_values = [STATUS_MAP['Available']] * len(plugin_times)
        ax.scatter(plugin_times, plugin_values, color='blue', marker='o', label='Plugin Events')

        # Setting the rest of the plot details
        yticklabels = STATUS_MAP.keys()
        yticks = [STATUS_MAP[x] for x in yticklabels]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        title = "Charger Usage Over Time"
        if 'charger_name' in kwargs:
            charger_name = kwargs['charger_name']
            title += f" at {charger_name}"
        ax.set_title(title)

        ax.set_xlabel("New Zealand local date and time")

        # Save the figure if output_path is provided
        if 'output_path' in kwargs:
            output_path = Path(kwargs['output_path'])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()

    finally:
        plt.close(fig)



def plot_hourly_demand(hourly_demand, figsize=FIGSIZE, **kwargs):
    """Area chart of hourly (inferred) demand per station"""
    tick_spacing = 0.96 # visually adjust plot for 24 hour steps
    ticks = [i * tick_spacing for i in range(25)]

    try:
        fig, ax = plt.subplots(figsize=figsize)

        # Ensure data is in expected format
        hourly_demand = hourly_demand.fillna(0)

        # Check if there is data to plot
        if hourly_demand[['q', 'r"', 'm', 'r']].sum().sum() == 0:
            ax.text(0.5, 0.5, 'No data available', fontsize=15, ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            hourly_demand['charged'] = hourly_demand['q'] + hourly_demand['r"']
            cumulative_charged_m = hourly_demand['charged'] + hourly_demand['m']

            # Ensure the plot is meaningful even with low data
            if hourly_demand['charged'].max() < 0.1 and hourly_demand['m'].max() < 0.1:
                ax.text(0.5, 0.5, 'Insufficient data for detailed plot', fontsize=15, ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.fill_between(hourly_demand.index, hourly_demand['charged'], cumulative_charged_m, label='Missed Plug-ins', alpha=0.5, color='red')
                ax.plot(hourly_demand.index, cumulative_charged_m, color='red', linewidth=2, alpha=0.8)

                ax.fill_between(hourly_demand.index, 0, hourly_demand['charged'], label='Observed Plug-ins', alpha=0.5, color='green')
                ax.plot(hourly_demand.index, hourly_demand['charged'], color='green', linewidth=2, alpha=0.8)

                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Plug-ins per hour')
                title = 'Charger Plug-in Frequencies by Time of Day'
                if 'charger_name' in kwargs:
                    charger_name = kwargs['charger_name']
                    title += f" at {charger_name}"
                ax.set_title(title)
                labels = [str(int(round(tick / tick_spacing))) for tick in ticks]

                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)

                ax.legend()

        ax.set_position([0.1, 0.1, 0.85, 0.85]) 
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax.set_xlim(left=0, right=ticks[-1])
        ax.set_ylim(bottom=0, top=1.5)

        if 'output_path' in kwargs:
            output_path = Path(kwargs['output_path'])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()

    finally:
        plt.close(fig)


# Main
        
start_time = time.time()

script_number = extract_script_number(__file__)

# Ensure DATADIR exists
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

# Determine input files
input_north_island_stations_path = find_data_file(INPUT_NORTH_ISLAND_STATIONS_FILENAME, DATADIR)
input_south_island_stations_path = find_data_file(INPUT_SOUTH_ISLAND_STATIONS_FILENAME, DATADIR)
input_charging_station_transitions_path = find_data_file(INPUT_CHARGING_STATION_TRANSITIONS_FILENAME, DATADIR)


print("Loading station location data...")
charging_station_locations = pd.concat(
    [pd.read_csv(input_north_island_stations_path, index_col='SiteId'),
     pd.read_csv(input_south_island_stations_path, index_col='SiteId')]
)
charging_station_locations = charging_station_locations.groupby(charging_station_locations.index).first()

print("Loading station availability data...")
with open(input_charging_station_transitions_path, 'rb') as f:
    (status, resampled, plugin_events, queued_events, hourly_percentages, implied_demand, site_stats, all_hourly_rates) = pickle.load(f)

total_stations = len(status)

# Wipe existing PNGs
print("Deleting existing PNG files...")
for png_file in glob.glob(f"{PNG_DIR}/*.png"):
    os.remove(png_file)

print("Existing PNG files deleted.")

for idx, site in enumerate(status):

    tr = status[site]
    sp = status[site].IsSuspicious
    pe = plugin_events[site]
    hp = hourly_percentages[site]
    id = implied_demand[site]
    st = site_stats[site]
    
    if len(tr) >= MIN_TRANSITIONS and site in charging_station_locations.index:

        charger_name = charging_station_locations.loc[site, 'Name']

        plot_evroam_availabilities(
            tr,
            pe,
            sp,
            charger_name=charger_name,
            output_path=f"{OUTPUTDIR}/png/charger_availabilities_{site}.png")

        plot_evroam_availability_pattern(
                hp,
                freq_to_hours(RESAMPLE_FREQUENCY),
                charger_name=charger_name,
                output_path=f"{OUTPUTDIR}/png/availability_pattern_{site}.png")

        plot_hourly_demand(
                id,
                charger_name=charger_name,
                output_path=f"{OUTPUTDIR}/png/hourly_demand_{site}.png")

    percent_complete = ((idx + 1) / total_stations) * 100
    sys.stdout.write(f"\r{percent_complete:.2f}% complete, working on station {site}" + " " * 50)
    sys.stdout.flush()

print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")