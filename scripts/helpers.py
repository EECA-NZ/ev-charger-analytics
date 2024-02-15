import os
import pandas as pd

def extract_script_number(script_name):
    base_name = os.path.basename(script_name)
    number = base_name.split('_')[0]
    return number

def find_data_file(lookup_name, data_directory):
    for file in os.listdir(data_directory):
        # Ignore the script number in the filename
        if file.endswith(lookup_name):
            return os.path.join(data_directory, file)
    return None

def generate_html_content():
    return """
<!DOCTYPE html>
<html>
<head>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden; 
        }
        .image-container {
            display: grid;
            grid-template-columns: repeat(1, 1fr);
            grid-template-rows: repeat(1, 1fr);
            gap: 10px;
            width: 100%;
            height: 100%;
            padding: 10px;
            box-sizing: border-box;
        }
        h1 {
            font-size: 10px;
            grid-column: 1 / -1;
            text-align: center;
            margin-bottom: 2px;
        }
        img {
            width: 100%;
            height: auto;
            object-fit: contain;
            display: block;
        }
        .no-data {
            text-align: center;
            color: grey;
            font-size: 16px;
            grid-column: 1 / -1;
        }
    </style>
    <script>
        window.onload = function() {
            // Parse the ChargingStationId and StationName from the URL
            const urlParams = new URLSearchParams(window.location.search);
            const ChargingStationId = urlParams.get('ChargingStationId');
            const StationName = urlParams.get('StationName');

            // Construct the image paths
            const chargerAvailabilitiesPath = `./png/charger_availabilities_${ChargingStationId}.png`;
            const availabilityPatternPath = `./png/availability_pattern_${ChargingStationId}.png`;
            const hourlyDemandPath = `./png/hourly_demand_${ChargingStationId}.png`;

            // Set the src attributes of the image tags
            document.getElementById("availabilityPatternImage").src = availabilityPatternPath;
            document.getElementById("chargerAvailabilitiesImage").src = chargerAvailabilitiesPath;
            document.getElementById("hourlyDemandImage").src = hourlyDemandPath;
            document.getElementById("stationStatsImage").src = stationStatsPath;

            // Update the header text with the Station Name
            document.getElementById("headerText").innerText = `Charging Station Usage Pattern for ${StationName}`;
        };
    </script>
</head>
<body>
    <div class="image-container">
        <img id="hourlyDemandImage" alt="Hourly Demand Pattern">
        <img id="availabilityPatternImage" alt="Availability Pattern">
        <img id="chargerAvailabilitiesImage" alt="Charger Availabilities">
        <div id="noDataText" class="no-data"></div> <!-- Placeholder for no data text -->
    </div>
</body>
</html>
"""


def freq_to_hours(freq_str):
    """
    Convert a frequency string to a number of hours.

    Parameters:
    - freq_str: A string representing the frequency (e.g., '10T' for 10 minutes).

    Returns:
    - A floating-point number representing the frequency in hours.
    """
    timedelta = pd.to_timedelta(freq_str)
    return timedelta.total_seconds() / 3600
