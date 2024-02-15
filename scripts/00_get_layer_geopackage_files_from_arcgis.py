from helpers import extract_script_number

script_number = extract_script_number(__file__)

print(
f"""
Placeholder. In the meantime, please manually place the following files:

../data/{script_number}_ChargingHubSites.gpkg
../data/{script_number}_StateHighway_view.gpkg
../data/{script_number}_SA2_year_popn_EV.gpkg
"""
)