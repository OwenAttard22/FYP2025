import os
import re
import tqdm
import random
import argparse
from datetime import datetime, timedelta
from AlphaScenarioGen import haversine  # Ensure this function is available in AlphaScenarioGen

# Directory paths
TRAIN = "./scenario/Charlie/TRAIN"
TEST = "./scenario/Charlie/TEST"
VAL = "./scenario/Charlie/VAL"

COOL_PLANE_NAMES = [
    'HAWK', 'EAGLE', 'FALCON', 'SCORPION', 'VIPER', 'RAVEN',
    'PHOENIX', 'SPARROW', 'HORNET', 'PEGASUS', 'TALON', 'GRYPHON',
    'WRAITH', 'LIGHTNING', 'DRAGON', 'THUNDERBIRD', 'STORM', 'BLADE'
]

# Scenario configuration
SCN_TIME = "00:00:00.00>"
NAUTICAL_MILE = 1852
R = 6371000
MIN_DISTANCE = 15000  # Minimum spacing between aircraft

P3_END_LAT = 36.692056
P3_END_LON = 20.453467

BASE_PATHS = [
    {"name": "P1", "start_lat": 38.764163, "start_lon": 9.858767, "end_lat": 36.644421, "end_lon": 15.111351},
    {"name": "P2", "start_lat": 34.298415, "start_lon": 10.082599, "end_lat": 36.644421, "end_lon": 15.111351}
]

def interpolate_point(start_lat, start_lon, end_lat, end_lon, fraction):
    lat = start_lat + fraction * (end_lat - start_lat)
    lon = start_lon + fraction * (end_lon - start_lon)
    return lat, lon

def generate_random_lat_long_within_percentage(path, percentage, existing_locations):
    """Ensure spawning maintains a minimum distance from existing planes."""
    while True:
        fraction = random.uniform(0, percentage)
        lat, lon = interpolate_point(path["start_lat"], path["start_lon"], path["end_lat"], path["end_lon"], fraction)
        if all(haversine(lat, lon, ex_lat, ex_lon) >= MIN_DISTANCE for ex_lat, ex_lon in existing_locations):
            return lat, lon

def get_next_index(directory, scenario_type):
    files = os.listdir(directory)
    pattern = re.compile(rf"{scenario_type}_(\d+).scn")
    max_index = 0
    for file in files:
        match = pattern.match(file)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1

def generate_scenarios(scenarios, scenario_type):
    directories = {"TRAIN": TRAIN, "TEST": TEST, "VAL": VAL}
    directory = directories.get(scenario_type.upper())
    if not directory:
        raise ValueError(f"Invalid scenario type: {scenario_type}")

    os.makedirs(directory, exist_ok=True)
    start_index = get_next_index(directory, scenario_type)

    for i in tqdm.tqdm(range(scenarios)):
        file_path = os.path.join(directory, f"{scenario_type}_{start_index + i:04d}.scn")
        plane_names = iter(COOL_PLANE_NAMES)
        existing_locations = []

        with open(file_path, "w") as file:
            file.write(f'{SCN_TIME}CRECMD COLOUR 255,255,255\n')
            file.write(f'{SCN_TIME}AREA 40.06,7.97 33.62,21.87\n')
            file.write(f'{SCN_TIME}LINE P3 {BASE_PATHS[0]["end_lat"]},{BASE_PATHS[0]["end_lon"]} {P3_END_LAT},{P3_END_LON}\n')
            for path in BASE_PATHS:
                file.write(f'{SCN_TIME}LINE {path["name"]} {path["start_lat"]},{path["start_lon"]} {path["end_lat"]},{path["end_lon"]}\n')
            file.write(f'{SCN_TIME}PAN 37, 14.24\n')
            file.write(f'{SCN_TIME}ZOOM 0.3\n')
            file.write(f'{SCN_TIME}TRAILS ON\n')
            # file.write(f'{SCN_TIME}VNAV OFF\n')

            while True:
                try:
                    for path in BASE_PATHS:
                        plane_name = next(plane_names)
                        lat, lon = generate_random_lat_long_within_percentage(path, 0.5, existing_locations)
                        existing_locations.append((lat, lon))
                        speed = random.randint(129, 319)
                        file.write(f'{SCN_TIME}CRE {plane_name}, B747, {lat},{lon}, 50, 10000, {speed}\n')
                        # file.write(f'{SCN_TIME}ADDWPT {plane_name}, {path["end_lat"]},{path["end_lon"]}\n')
                        # file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO DEST {plane_name}, {P3_END_LAT},{P3_END_LON}\n')
                        file.write(f'{SCN_TIME}ADDWPT {plane_name}, {path["end_lat"]},{path["end_lon"]}\n')
                        file.write(f'{SCN_TIME}DEST   {plane_name}, {P3_END_LAT},{P3_END_LON}\n')

                except StopIteration:
                    break

            file.write(f'{SCN_TIME}PLUGIN CharliePlugin\n')
            # file.write(f'{SCN_TIME}OP\n')
            # file.write(f'{SCN_TIME}FF\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create scenario files for Charlie Environment.")
    parser.add_argument("-s", type=int, help="Number of scenarios to generate", required=True)
    parser.add_argument("-t", type=str, help="Scenario type (TRAIN, TEST, VAL)", required=True)
    args = parser.parse_args()
    generate_scenarios(args.s, args.t)
