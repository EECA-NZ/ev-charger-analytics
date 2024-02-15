"""
Multi-charger-site rules:
- If any charger is available, the site is available.
- If all chargers are unavailable, the site is unavailable.
- Otherwise [i.e., if no charger is available and at least one is occupied], the site is occupied.

Single-charger-site rules:
- If any charger is available, the site is available. I.e. if the charger is available, the site is available.
- If all chargers are unavailable, the site is unavailable. I.e. if the charger is unavailable, the site is unavailable.
- Otherwise [i.e., if no charger is available and at least one is occupied], the site is occupied. I.e. if the charger is occupied, the site is occupied.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
from pandas import DataFrame, to_datetime
from ModernDataPlatform import DbInteract

pd.options.mode.chained_assignment = None

from helpers import extract_script_number, find_data_file

# Constants
from constants import DEV_OR_PRD, DATADIR, LOCALTIME, MAX_PLAUSIBLE_PERIOD_BETWEEN_TRANSITIONS, IQR_MULTIPLIER, RESAMPLE_FREQUENCY, EV_CAR_PERCENTAGES

INPUT_NORTH_ISLAND_STATIONS_FILENAME = 'north_island_stations_with_volumes_and_SA2.csv'
INPUT_SOUTH_ISLAND_STATIONS_FILENAME = 'south_island_stations_with_volumes_and_SA2.csv'
OUTPUT_TURN_IN_RATES_FILENAME = 'turn_in_rates_data.csv'
OUTPUT_CHARGING_STATION_TRANSITIONS_FILENAME = 'charging_station_transitions.gpickle'

def get_raw_data(env: str = DEV_OR_PRD, localtime: str = LOCALTIME):
    """
    Fetch raw data (all stations) from the GIDI database and join using pandas.
    The data is then converted to the specified local timezone.

    Parameters:
    - env: The environment setting (production, development, etc.) for database interaction.
    - localtime: The local timezone to convert the data to.

    Returns:
    - A pandas DataFrame containing the fetched data with times converted to the specified local timezone.
    """
    mdp_db = DbInteract(env=env)
    df = mdp_db.read('''
    SELECT CS.SiteId,
    A.*
    FROM EVRoam.Availabilities AS A
    INNER JOIN EVRoam.ChargingStations AS CS ON A.ChargingStationId = CS.ChargingStationId
    INNER JOIN EVRoam.Sites AS S ON CS.SiteId = S.SiteId
    WHERE CS.ProviderDeleted = 0
    AND S.ProviderDeleted = 0
    AND A.AvailabilityTime >= '2023-08-01'
    AND A.AvailabilityTime < '2023-11-01';
    ''')
    # Convert UTC times to NZST/NZDT, taking into account daylight saving
    df['LocalTimePeriod'] = df['AvailabilityTime'].dt.tz_localize('UTC').dt.tz_convert(localtime)
    return df


def deduplicate_station_data(df: DataFrame) -> DataFrame:
    """
    Clean and preprocess the raw dataframe for a single station by removing consecutive rows
    with the same AvailabilityStatus.

    Parameters:
    - df: The raw DataFrame containing station data.

    Returns:
    - A cleaned DataFrame with consecutive duplicate AvailabilityStatus rows removed.
    """
    # Sort by AvailabilityTime
    df = df.sort_values(by="AvailabilityTime")
    # Remove consecutive rows with the same AvailabilityStatus
    mask = df['AvailabilityStatus'] != df['AvailabilityStatus'].shift(1)
    df = df[mask]
    return df


def stats_time_in_each_status(df):
    """
    Calculate median and interquartile range of time spent in each availability status.

    Parameters:
    - df: DataFrame containing charging station data with AvailabilityTime and AvailabilityStatus columns.

    Returns:
    - A DataFrame with calculated median and interquartile range of time differences for each AvailabilityStatus.
    """
    df['AvailabilityTime'] = pd.to_datetime(df['AvailabilityTime'])
    df['NextAvailabilityTime'] = df.groupby('ChargingStationId')['AvailabilityTime'].shift(-1)
    df['TimeDiff'] = df['NextAvailabilityTime'] - df['AvailabilityTime']
   
    # Calculate median
    stats = df.groupby('AvailabilityStatus')['TimeDiff'].agg(['median'])

    # Calculate IQR 
    iqr_values = {}
    for status in df['AvailabilityStatus'].unique():
        group = df[df['AvailabilityStatus'] == status]['TimeDiff']
        iqr = group.quantile(0.75) - group.quantile(0.25)
        iqr_values[status] = iqr

    # Convert IQR values to DataFrame
    iqr_df = pd.DataFrame.from_dict(iqr_values, orient='index', columns=['iqr'])

    # Merge the IQR values back into the stats DataFrame
    stats = stats.merge(iqr_df, left_index=True, right_index=True)

    return stats



def flag_suspicious_transitions(df, iqr_multiplier: float, max_plausible_period_between_transitions: pd.Timedelta) -> pd.Series:
    """
    Flags transitions that are suspiciously long based on interquartile range and a maximum period.

    Parameters:
    - df: DataFrame containing charging station data.
    - iqr_multiplier: Number of IQRs used to flag long periods between transitions.
    - max_plausible_period_between_transitions: Maximum allowable period for a transition to be considered normal.

    Returns:
    - A boolean Series where True indicates a suspicious transition, indexed in increasing order.
    """
    _df = df.copy()
    # Calculate median and iqr in each status
    status_stats = stats_time_in_each_status(_df)
    # Calculate time until next transition
    _df['NextAvailabilityTime'] = _df.groupby('ChargingStationId')['AvailabilityTime'].shift(-1)
    _df['TimeToNextTransition'] = _df['NextAvailabilityTime'] - _df['AvailabilityTime']
    # Flag rows as suspicious
    def is_suspicious(row):
        median = status_stats.loc[row['AvailabilityStatus'], 'median']
        iqr = status_stats.loc[row['AvailabilityStatus'], 'iqr']
        upper_limit = median + (iqr_multiplier * iqr)
        return row['TimeToNextTransition'] > upper_limit and row['TimeToNextTransition'] > max_plausible_period_between_transitions
    return _df.apply(is_suspicious, axis=1).sort_index()


def process_cleaned_data(df: DataFrame, localtime: str = LOCALTIME) -> DataFrame:
    """
    Process the cleaned dataframe for a single station. Converts AvailabilityTime to local time,
    determines the next availability status, and selects relevant columns.

    Parameters:
    - df: DataFrame to process.
    - localtime: Local time zone for time conversion.

    Returns:
    - Processed DataFrame with local time period and next availability status.
    """
    # First, ensure data is sorted by AvailabilityTime
    df = df.sort_values(by="AvailabilityTime")
    # Convert AvailabilityTime to LocalTimePeriod in the specified timezone
    df['LocalTimePeriod'] = to_datetime(df['AvailabilityTime']).dt.tz_localize('UTC').dt.tz_convert(localtime)
    # Determine the next availability status
    df['Next_AvailabilityStatus'] = df['AvailabilityStatus'].shift(-1)
    # Ensure the DataFrame contains only relevant columns
    columns = ['LocalTimePeriod', 'SiteId', 'ChargingStationId', 'AvailabilityStatus', 'Next_AvailabilityStatus', 'IsSuspicious']
    return df[columns]


def resample_data(df: DataFrame, freq: str = 'H', localtime: str = LOCALTIME) -> DataFrame:
    """
    Resample the dataframe for a single station to represent availabilities based on the given frequency.

    Parameters:
    - df: DataFrame to resample.
    - freq: Frequency for resampling ('H' for hourly, '10T' for ten minutes, etc.).
    - localtime: Local time zone for time conversion.

    Returns:
    - A DataFrame resampled according to the specified frequency.
    """
    # Ensure data is sorted by LocalTimePeriod
    df = df.sort_values(by="LocalTimePeriod")
    start = df['LocalTimePeriod'].min().ceil(freq)
    end = df['LocalTimePeriod'].max().ceil(freq)
    # Create a range for the time period based on the given frequency
    range_period = pd.date_range(start=start, end=end, freq=freq, tz=localtime)
    # Use forward fill to get the availability status for each period
    df.set_index('LocalTimePeriod', inplace=True)
    resampled_data = df.reindex(range_period, method='ffill').reset_index()
    resampled_data = resampled_data.rename(columns={"index": "LocalTimePeriod"})
    # Handle potential NaNs after resampling
    resampled_data.dropna(subset=['AvailabilityStatus'], inplace=True)
    return resampled_data


def calculate_hourly_status_percentages(resampled_status):
    """
    Calculate hourly percentages of each availability status.

    Parameters:
    - resampled_status: DataFrame with resampled station status data.

    Returns:
    - A DataFrame with hourly percentages of each availability status.
    """
    # Ensure the data is sorted by LocalTimePeriod
    resampled_status = resampled_status.sort_values(by="LocalTimePeriod")
    # Extract the hour of the day from LocalTimePeriod
    resampled_status['HourOfDay'] = resampled_status['LocalTimePeriod'].dt.hour
    # Calculate the normalized value counts of AvailabilityStatus
    hourly_percentages = resampled_status.groupby('HourOfDay')['AvailabilityStatus'].value_counts(normalize=True).unstack().fillna(0) * 100
    # Reset index to make HourOfDay a column
    hourly_percentages = hourly_percentages.reset_index()
    return hourly_percentages


def calculate_unique_days(filtered_data):
    """
    Calculate the number of unique days in the dataset.

    Parameters:
    - filtered_data: DataFrame containing data with a LocalTimePeriod column.

    Returns:
    - An integer representing the number of unique days in the dataset.
    """
    if filtered_data.empty or 'LocalTimePeriod' not in filtered_data.columns:
        return 0
    unique_dates = filtered_data['LocalTimePeriod'].dt.date.unique()
    return len(unique_dates)


def calculate_hourly_rates(fractional_availability, actual_unique_days, queued_counts, plugin_counts, site_id):
    """
    Calculate the hourly rates of queue, free charger arrival, missed arrivals, and true arrival rates for an EV charging station.
    
    The function uses observed plugin rates and inferred queuing information to estimate true arrival rates, accounting for
    the times when a station is busy and a vehicle might have driven on without charging. It also calculates the missed rates
    based on the maximum of observed empty arrivals to true arrival rates and observed arrival rates.

    Parameters:
    - fractional_availability (Series): The fraction of time the charging station was available for each hour of the day.
    - actual_unique_days (int): The total number of unique days in the dataset.
    - queued_counts (Series): The inferred count of vehicles that queued for each hour of the day.
    - plugin_counts (Series): The observed count of vehicles that were charging for each hour of the day.
    - station_id (str): Identifier for the charging station.

    Returns:
    - DataFrame: A DataFrame with the following columns:
        - t (hour of day)
        - N (number of days)
        - a (fractional availability)
        - O (observed plugin rate)
        - Q (inferred to have queued)
        - R" (inferred arrivals that didn't queue)
        - R (inferred true arrivals)
        - M (inferred to have driven on, missed)
        - q (queue rate per hour)
        - r" (free charger arrival rate per hour)
        - m (missed rate per hour)
        - r (true arrival rate per hour)
        - ChargingStationId (identifier for the charging station)
    """
    fractional_availability_1d = fractional_availability['Available']

    # Initialize the DataFrame with the raw data and calculated columns
    df = pd.DataFrame({
        't': range(24),  # hour of day
        'N': actual_unique_days,  # number of days
        'a': fractional_availability_1d,  # fractional availability
        'O': plugin_counts.fillna(0),  # observed plugin rate with NaNs filled as 0
        'Q': queued_counts.reindex(range(24), fill_value=0),  # Ensure queued_counts has 24 entries
    })
    
    # Calculate R", R, and M
    df['R"'] = df['O'] - df['Q']
    df['R'] = np.where(df['a'] == 0, df['O'], np.maximum(df['R"'] / df['a'], df['O']))
    df['M'] = df['R'] - df['Q'] - df['R"']
    
    # Calculate 'q', 'r"', 'm', 'r'
    df['q'] = df['Q'] / df['N']
    df['r"'] = df['R"'] / df['N']
    df['m'] = df['M'] / df['N']
    df['r'] = df['q'] + df['r"'] + df['m']
    
    # Add SiteId
    df['SiteId'] = site_id
    
    return df

def fractional_availability_by_hour_of_day(status_percentages):
    # Check if 'Available' column exists in the status_percentages DataFrame
    if 'Available' in status_percentages.columns:
        fractional_availability = status_percentages[['HourOfDay', 'Available']].copy()
        fractional_availability['Available'] = fractional_availability['Available'] / 100
        fractional_availability.set_index('HourOfDay', inplace=True)
    else:
        fractional_availability = pd.DataFrame({'Available': [0]*24}, index=range(24))
    return fractional_availability

def plugin_count_by_hour_of_day(plugin_events):
    if 'LocalTimePeriod' in plugin_events.columns:
        plugin_events['LocalHour'] = plugin_events['LocalTimePeriod'].dt.hour
        # Group by 'LocalHour' to get counts of queued transitions
        plugin_counts = plugin_events.groupby('LocalHour').size()
    else:
        plugin_counts = pd.Series([0] * 24, index=pd.Index(range(24), name='LocalHour'))
    # Ensure there is an entry for every hour of the day
    plugin_counts = plugin_counts.reindex(range(24), fill_value=0)
    return plugin_counts

def calculate_hourly_demand(site_plugin_events, queued_plugin_events, status_percentages, unique_days, site_id):
    """
    Calculate hourly demand statistics for an EV charging site.

    Parameters:
    - site_plugin_events: DataFrame with plugin event data for the site.
    - queued_plugin_events: DataFrame with queued plugin event data for the site.
    - status_percentages: DataFrame with status percentages for the site.
    - unique_days: Integer representing the number of unique days in the dataset.
    - site_id: The identifier for the site.

    Returns:
    - A DataFrame containing hourly demand statistics for the site.
    """
    fractional_availability = fractional_availability_by_hour_of_day(status_percentages)
    plugin_counts = plugin_count_by_hour_of_day(site_plugin_events)
    queued_counts = plugin_count_by_hour_of_day(queued_plugin_events)
    return calculate_hourly_rates(fractional_availability, unique_days, queued_counts, plugin_counts, site_id)

def median_occupied_duration(site_state_raw):
    """
    Calculate the median duration for 'Occupied' status for each charging station.

    Parameters:
    - site_state_raw: DataFrame containing raw site state data.

    Returns:
    - A timedelta object representing the median duration of the 'Occupied' status.
    """
    # Ensure data is sorted by 'LocalTime'
    site_state_raw = site_state_raw.sort_values(by="LocalTimePeriod")
    # Calculate time to next transition for each row
    site_state_raw['NextAvailabilityTime'] = site_state_raw['LocalTimePeriod'].shift(-1)
    site_state_raw['event_duration'] = site_state_raw['NextAvailabilityTime'] - site_state_raw['LocalTimePeriod']
    # Filter for 'Occupied' status
    occupied_status = site_state_raw[site_state_raw['AvailabilityStatus'] == 'Occupied']
    # Calculate median duration for each ChargingStationId
    median_duration = occupied_status['event_duration'].median()
    if pd.isna(median_duration):
        median_duration = pd.Timedelta(seconds=0)
    return median_duration

def duration_to_string(duration):
    """
    Convert a pandas Timedelta duration to a presentable string format.

    Parameters:
    - duration: A pandas Timedelta object representing a duration.

    Returns:
    - A string representation of the duration in 'HH:MM:SS' format, or 'N/A' if the duration is not valid.
    """
    if not pd.isna(duration):
        duration_str = f"{pd.Timedelta(duration).components.hours:02d}:{pd.Timedelta(duration).components.minutes:02d}:{pd.Timedelta(duration).components.seconds:02d}"
    else:
        duration_str = 'N/A'
    return duration_str


def calculate_peak_times(hourly_demand, ratio_to_peak=2.5):
    """
    Calculate peak times of day for each charging station based on 'Occupied' status.

    Parameters:
    - hourly_demand: DataFrame containing hourly demand data.
    - ratio_to_peak: The ratio used to define peak times (default is 2.5).

    Returns:
    - A string representing the peak start and end times in 'HH:00-HH:00' format.
    """
    threshold = hourly_demand.r.max()/ratio_to_peak
    peak_start = hourly_demand[hourly_demand.r > threshold].t.min()
    peak_end = hourly_demand[hourly_demand.r > threshold].t.max()
    return str(peak_start).zfill(2) + ":00-" + str(peak_end).zfill(2) + ":00"


def calculate_evs_supported(average_daily_charges, F=0.15, Ce=0.4):
    """
    Calculate the number of EVs supported by each site per day.

    Parameters:
    - average_daily_charges: Average daily charges at the site.
    - F: Fraction of charging sessions that occur on public chargers (default is 0.15).
    - Ce: Average frequency of charging for an individual EV per day (default is 0.4).

    Returns:
    - The approximate number of EVs supported at that charger per day.
    """
    # Calculate EVs supported by one charger per day
    return (average_daily_charges / F / Ce).round().astype(int)


def calculate_total_national_benefit(evs_supported, B=15000):
    """
    Calculate the total national benefit based on the number of EVs supported by each charger.

    Parameters:
    - evs_supported: The number of EVs supported per charger per day.
    - B: Lifetime NPV of purchasing an EV rather than an ICE vehicle in 2025 (default is $15,000).

    Returns:
    - The total national benefit as an integer.
    """
    # Calculate Total National Benefit (N)
    return (B * evs_supported).round().astype(int)


def cost_of_missed_charges(missed_charges_per_day, B=15000, F=0.15, Ce=0.4):
    """
    Calculate the foregone benefit due to missed charges for a single site.

    Parameters:
    - missed_charges_per_day: Average missed charges per day for a station.
    - B: Lifetime NPV of purchasing an EV rather than an ICE vehicle in 2025 (default is $15,000).
    - F: Fraction of charging sessions that occur on public chargers (default is 0.15).
    - Ce: Average frequency of charging for an individual EV per day (default is 0.4).

    Returns:
    - A formatted string representing the cost of missed charges.
    """
    cost = int(B * (missed_charges_per_day / F) * (1 / Ce))
    return f"${cost:,}"


def calculate_queuing_time_value_loss_per_year(median_duration, queued_charges_per_day, travel_rate=0.67):
    """
    Calculate the annual value loss due to cars queued at a charging station.

    Parameters:
    - median_duration: Median duration of a charging session.
    - queued_charges_per_day: Total count of queued cars per day at the station.
    - travel_rate: Cost rate per minute per person (default is $0.67, i.e., $40/h/vehicle).

    Returns:
    - The total annual value loss in dollars due to queuing at the station.
    """
    return median_duration / 2 / timedelta(minutes=1) * queued_charges_per_day * travel_rate * 365


def merge_traffic_and_demand_data_with_station_data(all_hourly_rates, all_stations, year=2023):
    """
    Loads and combines station data with hourly rates, then calculates average daily metrics
    including observed plug-ins and missed charges. It computes the daily demand, turn-in rate,
    and EV-specific turn-in rate for each station based on the given year's EV percentage.

    Parameters:
    - all_hourly_rates: DataFrame containing hourly rates data for all stations.
    - all_stations: DataFrame containing station data.
    - year: The year for which to use the EV car percentage (defaults to 2023).

    Returns:
    - A merged DataFrame with additional calculated metrics.
    """
    # Aggregate values for each SiteId
    avg_values = all_hourly_rates.groupby('SiteId').agg({'m':'sum', 'O':'sum'}).reset_index()
    number_of_days_observed_per_station = all_hourly_rates.groupby('SiteId')['N'].first().reset_index()
    number_of_days_observed_per_station.rename(columns={'N': 'Days_observed'}, inplace=True)
    # Merge the station data with aggregated values
    final_merged_data = pd.merge(all_stations, avg_values, on='SiteId', how='left')
    final_merged_data = pd.merge(final_merged_data, number_of_days_observed_per_station, on='SiteId', how='left')
    # Calculate additional columns
    final_merged_data['avg_observed_plugins_per_day'] = final_merged_data['O'] / final_merged_data['Days_observed']
    final_merged_data.rename(columns={'m': 'avg_misses_per_day'}, inplace=True)
    final_merged_data['daily_demand'] = final_merged_data['avg_observed_plugins_per_day'] + final_merged_data['avg_misses_per_day']
    final_merged_data['turnin_rate'] = final_merged_data['daily_demand'] / final_merged_data['trafficVolume']
    # Use the EV percentage for the specified year 
    ev_percentage = EV_CAR_PERCENTAGES.get(year, None)
    final_merged_data['EV_turnin_rate'] = final_merged_data['daily_demand'] / (final_merged_data['trafficVolume'] * ev_percentage)
    return final_merged_data


def get_plugin_events(df):
    """
    Filter rows to identify when a charging station changes status from not 'Occupied' to 'Occupied'.

    Parameters:
    - df: DataFrame with columns 'ChargingStationId', 'AvailabilityStatus', etc.

    Returns:
    - DataFrame with rows where the previous AvailabilityStatus was not "Occupied" and the current AvailabilityStatus is "Occupied".
    """
    temp_df = df.copy()
    temp_df['PrevAvailabilityStatus'] = temp_df.groupby('ChargingStationId')['AvailabilityStatus'].shift(1)
    plugin_events = temp_df[(temp_df['PrevAvailabilityStatus'] != 'Occupied') & (temp_df['AvailabilityStatus'] == 'Occupied')]
    return plugin_events


def get_queued_plugin_events(site_plugin_events, site_state_raw, transition_time_threshold=pd.Timedelta(minutes=5)):
    """
    Identify queued plugin events based on transition time thresholds.

    Parameters:
    - site_plugin_events: DataFrame containing plugin events for a site.
    - site_state_raw: DataFrame containing raw site state data.
    - transition_time_threshold: The time threshold for considering an event as queued (default is 5 minutes).

    Returns:
    - A DataFrame containing queued plugin events.
    """
    queued_plugin_events = pd.DataFrame()
    for index, event in site_plugin_events.iterrows():
        plugin_time = event['LocalTimePeriod']
        last_states = site_state_raw[
            (site_state_raw['LocalTimePeriod'] < plugin_time)
        ].tail(2)
        if last_states.shape[0] == 2 and last_states.iloc[1]['AvailabilityStatus'] == 'Available' and last_states.iloc[0]['AvailabilityStatus'] != 'Available':
            if (plugin_time - last_states.iloc[1]['LocalTimePeriod']) <= transition_time_threshold:
                queued_plugin_events = pd.concat([queued_plugin_events, pd.DataFrame([event])], ignore_index=True)
    return queued_plugin_events


def get_site_timeseries(raw_data, site_id,
                        localtime=LOCALTIME,
                        resample_freq=RESAMPLE_FREQUENCY,
                        iqr_multiplier=IQR_MULTIPLIER,
                        max_plausible_period_between_transitions=MAX_PLAUSIBLE_PERIOD_BETWEEN_TRANSITIONS):
    """
    Generate a timeseries dataset for a specific site including resampled status data and plugin events.

    Parameters:
    - raw_data: DataFrame containing the raw data for all sites.
    - site_id: The identifier of the site to process.
    - localtime: Local time zone for time conversion.
    - resample_freq: Frequency for resampling data.
    - iqr_multiplier: Multiplier for interquartile range to determine suspicious transitions.
    - max_plausible_period_between_transitions: Maximum allowable period for a transition to be considered normal.

    Returns:
    - A tuple containing site state raw data, resampled data, and plugin events.
    """
    events_of_status = pd.DataFrame()
    resampled_status = pd.DataFrame()
    site_plugin_events = pd.DataFrame()
    site_station_status = raw_data[raw_data['SiteId'] == site_id]
    for station_id in site_station_status.ChargingStationId.unique():
        station_status = site_station_status[site_station_status['ChargingStationId'] == station_id]
        # Remove successive rows with identical status
        station_status = deduplicate_station_data(station_status)
        # Determine filter: which transitions will have the time period after them ignored
        station_status['IsSuspicious'] = flag_suspicious_transitions(station_status, iqr_multiplier, max_plausible_period_between_transitions)
        # Identify plugin events
        station_plugin_events = get_plugin_events(station_status)
        # Resample station data
        resampled_station_status = resample_data(station_status, freq=resample_freq, localtime=localtime)
        # Append the data for this station
        events_of_status = pd.concat([events_of_status, station_status])
        resampled_status = pd.concat([resampled_status, resampled_station_status])
        site_plugin_events = pd.concat([site_plugin_events, station_plugin_events])
    site_plugin_events = site_plugin_events[site_plugin_events['IsSuspicious'] == False]
    site_data = multiindex_by_period_and_station(events_of_status)
    site_data = forward_fill_site_data(site_data)
    resampled_site_data = multiindex_by_period_and_station(resampled_status)
    site_state_raw = site_data.groupby('LocalTimePeriod').apply(determine_site_status).to_frame(name='AvailabilityStatus')
    site_state_raw['IsSuspicious'] = site_data.groupby('LocalTimePeriod').apply(determine_site_suspicious).to_frame(name='IsSuspicious')
    site_state_res = resampled_site_data.groupby('LocalTimePeriod').apply(determine_site_status).to_frame(name='AvailabilityStatus')
    site_state_res['IsSuspicious'] = resampled_site_data.groupby('LocalTimePeriod').apply(determine_site_suspicious).to_frame(name='IsSuspicious')
    site_state_res['SiteId'] = site_id
    site_state_raw['SiteId'] = site_id
    site_state_raw = site_state_raw.reset_index()
    site_state_res = site_state_res.reset_index()
    return site_state_raw, site_state_res, site_plugin_events


def add_counts(site_data):
    """
    Add count columns to the site data DataFrame, representing the number of stations
    in each availability status ('Available', 'Occupied', and 'Unavailable') for each time period.

    Parameters:
    - site_data: DataFrame containing the site data with 'LocalTimePeriod' and 'AvailabilityStatus' columns.

    Returns:
    - A modified DataFrame with added columns for each availability status count and the total count.
    """
    site_data.reset_index(inplace=True)
    # Calculate the number of stations available, occupied, and unavailable for each time period
    status_counts = site_data.pivot_table(index='LocalTimePeriod',
                                            columns='AvailabilityStatus',
                                            aggfunc='size',
                                            fill_value=0)
    status_counts = status_counts.astype(int)
    status_counts['TotalCount'] = status_counts.sum(axis=1)
    site_data = site_data.join(status_counts, on='LocalTimePeriod')
    return site_data


def multiindex_by_period_and_station(resampled_status):
    """
    Create a DataFrame with a MultiIndex consisting of time periods and station IDs,
    containing availability status and suspicious flag for each station at each time period.

    Parameters:
    - resampled_status: DataFrame with resampled status data containing 'LocalTimePeriod',
      'ChargingStationId', 'AvailabilityStatus', and 'IsSuspicious' columns.

    Returns:
    - A DataFrame with a MultiIndex ('LocalTimePeriod', 'ChargingStationId') and columns
      for 'AvailabilityStatus' and 'IsSuspicious'.
    """
    # Create a multiindex for the resampled data
    index = pd.MultiIndex.from_product([resampled_status.LocalTimePeriod.unique(),
                                        resampled_status.ChargingStationId.unique()],
                                       names=['LocalTimePeriod', 'ChargingStationId'])
    columns = ['AvailabilityStatus', 'IsSuspicious']
    resampled_site_data = pd.DataFrame(index=index, columns=columns)
    for station_id in resampled_status.ChargingStationId.unique():
        station_status = resampled_status[resampled_status['ChargingStationId'] == station_id]
        # Update the MultiIndex DataFrame
        for time_period in station_status.LocalTimePeriod.unique():
            for column in columns:
                value = station_status.loc[station_status.LocalTimePeriod == time_period, column].values[0]
                resampled_site_data.loc[(time_period, station_id), column] = value
    return resampled_site_data


def determine_site_status(group):
    """
    Determine the overall status of a site based on the status of its charging stations
    for a specific time period.

    Parameters:
    - group: A group of rows from a DataFrame representing data for different stations at the same time period.

    Returns:
    - A string representing the site status ('Available', 'Occupied', or 'Unavailable').
    """
    if (group['AvailabilityStatus'] == 'Available').any():
        return 'Available'
    elif (group['AvailabilityStatus'] == 'Unavailable').all():
        return 'Unavailable'
    else:
        return 'Occupied'


def determine_site_suspicious(group):
    """
    Determine if a site's status is suspicious at a given time period by checking
    the suspicious flags of its individual stations.

    Parameters:
    - group: A group of rows from a DataFrame representing data for different stations at the same time period.

    Returns:
    - A boolean value indicating whether any of the stations at the site have a suspicious status for the time period.
    """
    return (group['IsSuspicious']).any()


def forward_fill_site_data(df):
    """
    Apply forward fill to each ChargingStationId in the site data.

    Parameters:
    - df: DataFrame with MultiIndex (LocalTimePeriod, ChargingStationId) and columns 'AvailabilityStatus' and 'IsSuspicious'

    Returns:
    - DataFrame with forward-filled data and the same MultiIndex structure.
    """
    # Apply forward fill within each group of ChargingStationId and reset the index within the lambda function
    filled_df = df.groupby(level='ChargingStationId').apply(lambda group: group.ffill().reset_index(level='ChargingStationId', drop=True))
    # Swap the levels of the MultiIndex and sort the index
    filled_df = filled_df.swaplevel('ChargingStationId', 'LocalTimePeriod')
    filled_df.sort_index(inplace=True)
    return filled_df


def get_evroam_charger_status(env: str = DEV_OR_PRD,
                              resample_freq: str=RESAMPLE_FREQUENCY,
                              localtime: str = LOCALTIME,
                              max_plausible_period_between_transitions: pd.Timedelta = MAX_PLAUSIBLE_PERIOD_BETWEEN_TRANSITIONS,
                              iqr_multiplier: float = IQR_MULTIPLIER) -> dict:
    """
    Pipeline to get, clean and resample charger status for all stations.
    Returns a dictionary with station IDs as keys and tuples of cleaned_data, 
    suspicious_mask, and resampled_data as values.
    Parameters:
    - env: Environment to use for database interaction.
    - resample_freq: Frequency for resampling data (e.g. '10T').
    - localtime: Local time zone for time conversion.
    - max_plausible_period_between_transitions: Maximum allowable period for a transition.
    - iqr_multiplier: Multiplier for interquartile to determine suspicious transitions.
    Returns:
    - A dictionary with charger status information.
    """
    raw_data = get_raw_data(env, localtime=localtime)
 
    status = {}
    resampled = {}
    plugin_events = {}
    queued_events = {}
    hourly_percentages = {}
    implied_demand = {}
    all_hourly_rates = {}
    site_stats = {}
    for site_id in raw_data.SiteId.unique():
        print(site_id)
        site_state_raw, site_state_res, site_plugin_events = get_site_timeseries(
            raw_data,
            site_id,
            localtime=localtime,
            resample_freq=resample_freq,
            iqr_multiplier=iqr_multiplier,
            max_plausible_period_between_transitions=max_plausible_period_between_transitions)

        queued_plugin_events = get_queued_plugin_events(site_plugin_events, site_state_raw)
        status_percentages = calculate_hourly_status_percentages(site_state_res)
        unique_days = calculate_unique_days(site_state_raw)
        hourly_demand = calculate_hourly_demand(site_plugin_events, queued_plugin_events, status_percentages, unique_days, site_id)

        active_station_count = len(site_plugin_events.ChargingStationId.unique())
        daily_observed_charges = (hourly_demand.O).sum()/unique_days
        daily_missed_charges = hourly_demand.m.sum()
        queued_charges_per_day = hourly_demand.q.sum()
        median_duration = median_occupied_duration(site_state_raw)
        peak_times_str = calculate_peak_times(hourly_demand, ratio_to_peak=2.5)
        evs_supported = calculate_evs_supported(daily_observed_charges)
        national_benefit = calculate_total_national_benefit(evs_supported)
        foregone_benefit = cost_of_missed_charges(daily_missed_charges)
        queue_time_cost = calculate_queuing_time_value_loss_per_year(median_duration, queued_charges_per_day, travel_rate=0.67)

        # Collate results into site dictionaries
        site_stats[site_id] = {
            'Active Stations': active_station_count,
            'Average Daily Charges': round(daily_observed_charges, 1) if daily_observed_charges else 0,
            'Queued Daily Charges': round(queued_charges_per_day, 1) if queued_charges_per_day else 0,
            'Missed Daily Charges': round(daily_missed_charges, 1) if daily_missed_charges else 0,
            'Charge Time Average': duration_to_string(median_duration),
            'Peak Times of Day': peak_times_str,
            'Estimated Vehicles Supported': f"~{evs_supported} EVs",
            'Social Benefit from EV Fleet': f"${national_benefit:,}",
            'Social Benefit foregone due to congestion': foregone_benefit,
            'Annual Queuing Cost': f"${int(round(queue_time_cost)):,}" if queue_time_cost else "0"
        }
        all_hourly_rates[site_id] = hourly_demand
        # Append to the all_hourly_rates DataFrame
        status[site_id] = site_state_raw
        resampled[site_id] = site_state_res
        plugin_events[site_id] = site_plugin_events
        queued_events[site_id] = queued_plugin_events
        hourly_percentages[site_id] = status_percentages
        implied_demand[site_id] = hourly_demand

    return status, resampled, plugin_events, queued_events, hourly_percentages, implied_demand, site_stats, all_hourly_rates



#### Main ####

start_time = time.time()

script_number = extract_script_number(__file__)

# Ensure DATADIR exists
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

print("Processing site station availability data...")
status, resampled, plugin_events, queued_events, hourly_percentages, implied_demand, site_stats, all_hourly_rates = get_evroam_charger_status(
    DEV_OR_PRD,
    resample_freq=RESAMPLE_FREQUENCY,
    localtime=LOCALTIME,
    max_plausible_period_between_transitions=MAX_PLAUSIBLE_PERIOD_BETWEEN_TRANSITIONS,
    iqr_multiplier=IQR_MULTIPLIER
)

hourly_rates_df = pd.concat(all_hourly_rates.values(), ignore_index=True)


# Determine input files
input_north_island_stations_path = find_data_file(INPUT_NORTH_ISLAND_STATIONS_FILENAME, DATADIR)
input_south_island_stations_path = find_data_file(INPUT_SOUTH_ISLAND_STATIONS_FILENAME, DATADIR)

# Load the data from the CSV files
print("Loading station location data...")
all_stations = pd.concat(
    [pd.read_csv(input_north_island_stations_path),
     pd.read_csv(input_south_island_stations_path)]
    )

# Dataframe for turn-in rate
merged_station_data = merge_traffic_and_demand_data_with_station_data(hourly_rates_df, all_stations)
csv_file_path = os.path.join(DATADIR, f'{script_number}_{OUTPUT_TURN_IN_RATES_FILENAME}')
merged_station_data.to_csv(csv_file_path)

with open(f'{DATADIR}/{script_number}_{OUTPUT_CHARGING_STATION_TRANSITIONS_FILENAME}', 'wb') as f:
    pickle.dump((status, resampled, plugin_events, queued_events, hourly_percentages, implied_demand, site_stats, all_hourly_rates), f, pickle.HIGHEST_PROTOCOL)

print("\nDone!")
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Script executed in: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:05.2f} seconds")