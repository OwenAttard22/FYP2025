import os
import re
import tqdm
import random
import argparse
import math
from datetime import datetime, timedelta

# Directory paths
TRAIN = "./scenario/Charlie/TRAIN"
TEST = "./scenario/Charlie/TEST"
VAL = "./scenario/Charlie/VAL"

COOL_PLANE_NAMES = [
    'HAWK', 'EAGLE', 'FALCON', 'SCORPION', 'VIPER', 'RAVEN',
    'PHOENIX', 'SPARROW', 'HORNET', 'PEGASUS', 'TALON', 'GRYPHON',
    'WRAITH', 'LIGHTNING', 'DRAGON', 'THUNDERBIRD', 'STORM', 'BLADE'
]
# Path coordinates
PATHS = [
    {"name": "P1", "start_lat": 38.764163, "start_lon": 9.858767, "end_lat": 36.644421, "end_lon": 15.111351},
    {"name": "P2", "start_lat": 34.298415, "start_lon": 10.082599, "end_lat": 36.644421, "end_lon": 15.111351}
]

P3_END_LAT = 36.692056
P3_END_LON = 20.453467

# Scenario configuration
SCN_TIME = "00:00:00.00>"
NAUTICAL_MILE = 1852
R = 6371000  # Radius of the Earth in meters


def interpolate_point(start_lat, start_lon, end_lat, end_lon, fraction):
    """Interpolate a point between the start and end based on a fraction (0 to 1)."""
    lat = start_lat + fraction * (end_lat - start_lat)
    lon = start_lon + fraction * (end_lon - start_lon)
    return lat, lon


def generate_random_lat_long_within_percentage(path, percentage):
    """Generate a random lat-long within the first 'percentage' of a path."""
    fraction = random.uniform(0, percentage)
    return interpolate_point(path["start_lat"], path["start_lon"], path["end_lat"], path["end_lon"], fraction)


def get_next_index(directory, scenario_type):
    """Get the next available scenario index."""
    files = os.listdir(directory)
    max_index = 0
    
    # Regular expression to match the file pattern
    pattern = re.compile(rf"{scenario_type}_(\d+).scn")
    
    for file in files:
        match = pattern.match(file)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    
    return max_index + 1  


def generate_scenarios(scenarios, scenario_type):
    """Generate scenario files."""
    directories = {"TRAIN": TRAIN, "TEST": TEST, "VAL": VAL}
    directory = directories.get(scenario_type.upper())
    
    if not directory:
        raise ValueError(f"Invalid scenario type: {scenario_type}. Use TRAIN, TEST, or VAL.")
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    start_index = get_next_index(directory, scenario_type)
    time = datetime.strptime("00:00:00.00", "%H:%M:%S.%f")
    time_delta = timedelta(minutes=3)

    # Flatten plane names list for sequential assignment
    plane_names = iter(COOL_PLANE_NAMES)

    for i in tqdm.tqdm(range(scenarios)):
        file_path = os.path.join(directory, f"{scenario_type}_{start_index + i:04d}.scn")

        with open(file_path, "w") as file:
            # Write static commands
            file.write(f'{SCN_TIME}CRECMD COLOUR 255,255,255\n')
            file.write(f'{SCN_TIME}AREA 40.06,7.97 33.62,21.87\n')
            file.write(f'{SCN_TIME}LINE P3 {PATHS[0]["end_lat"]},{PATHS[0]["end_lon"]} {P3_END_LAT},{P3_END_LON}\n')
            for path in PATHS:
                file.write(f'{SCN_TIME}LINE {path["name"]} {path["start_lat"]},{path["start_lon"]} {path["end_lat"]},{path["end_lon"]}\n')
            file.write(f'{SCN_TIME}PAN 37, 14.24\n')
            file.write(f'{SCN_TIME}ZOOM 0.3\n')
            file.write(f'{SCN_TIME}TRAILS ON\n')

            # Create planes and destinations
            while True:
                try:
                    for path in PATHS:
                        plane_name = next(plane_names)
                        lat, lon = generate_random_lat_long_within_percentage(path, 0.3)  # Spawn within first 30% of the path
                        time_str = time.strftime("%H:%M:%S.%f")[:-3]
                        file.write(f'{time_str}>CRE {plane_name}, B747, {lat},{lon}, 50, 10000, 300\n')
                        file.write(f'{time_str}>ADDWPT {plane_name}, {path["end_lat"]},{path["end_lon"]}\n')
                        file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO DEST {plane_name}, {P3_END_LAT},{P3_END_LON}\n')
                    time += time_delta  # Increment time after each set of planes
                except StopIteration:
                    break  # Stop when all plane names are used

            # Final commands
            file.write(f'{SCN_TIME}OP\n')
            file.write(f'{SCN_TIME}FF\n')


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Create scenario files for Alpha Environment.")
    parser.add_argument("-s", type=int, help="Number of scenarios to be generated", required=True)
    parser.add_argument("-t", type=str, help="Type of scenario to be generated (TRAIN, TEST, VAL)", required=True)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate scenarios
    generate_scenarios(args.s, args.t)
