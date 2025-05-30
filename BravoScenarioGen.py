import os
import re
import tqdm
import random
import argparse
import math
from datetime import datetime, timedelta
from AlphaScenarioGen import haversine

# Directory paths
TRAIN = "./scenario/Bravo/TRAIN"
TEST = "./scenario/Bravo/TEST"
VAL = "./scenario/Bravo/VAL"

COOL_PLANE_NAMES = [
    'HAWK', 'EAGLE', 'FALCON', 'SCORPION', 'VIPER', 'RAVEN',
    'PHOENIX', 'SPARROW', 'HORNET', 'PEGASUS', 'TALON', 'GRYPHON',
    'WRAITH', 'LIGHTNING', 'DRAGON', 'THUNDERBIRD', 'STORM', 'BLADE'
]

# Scenario configuration
SCN_TIME = "00:00:00.00>"
NAUTICAL_MILE = 1852
R = 6371000  # Radius of the Earth in meters

MIN_DISTANCE = 15000

def generate_dynamic_paths2():
    """Generate randomized paths for P1 and P2 with variable ending latitude."""
    lat_min = 36.5296
    lat_max = 38.20965

    end_lat_p1 = random.uniform(lat_min, lat_max)
    end_lat_p2 = random.uniform(lat_min, lat_max)

    return [
        {"name": "P1", "start_lat": 38.20965, "start_lon": 8.261966, "end_lat": end_lat_p1, "end_lon": 20.356376},
        {"name": "P2", "start_lat": 36.534355, "start_lon": 8.261966, "end_lat": end_lat_p2, "end_lon": 20.356376},
        {"name": "P3", "start_lat": 33.777445, "start_lon": 14.923951, "end_lat": 38.548577, "end_lon": 14.92395}
    ]

def get_spawnpoint(start_lat, start_lon, end_lat, end_lon, fraction):
    """Interpolate a point between the start and end based on a fraction (0 to 1)."""
    lat = start_lat + fraction * (end_lat - start_lat)
    lon = start_lon + fraction * (end_lon - start_lon)
    return lat, lon


def generate_random_lat_long_within_percentage(path, percentage, existing_locations):
    """Generate a spawn point within `percentage` of a path, maintaining min distance from existing."""
    while True:
        fraction = random.uniform(0, percentage)
        lat = path["start_lat"] + fraction * (path["end_lat"] - path["start_lat"])
        lon = path["start_lon"] + fraction * (path["end_lon"] - path["start_lon"])

        if all(haversine(lat, lon, ex_lat, ex_lon) >= MIN_DISTANCE for ex_lat, ex_lon in existing_locations):
            return lat, lon


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

    for i in tqdm.tqdm(range(scenarios)):
        file_path = os.path.join(directory, f"{scenario_type}_{start_index + i:04d}.scn")
        
        plane_names = iter(COOL_PLANE_NAMES)
        
        paths = generate_dynamic_paths2()

        with open(file_path, "w") as file:
            # Write static commands
            file.write(f'{SCN_TIME}CRECMD COLOUR 255,255,255\n')
            file.write(f'{SCN_TIME}AREA 40.06,7.97 33.62,21.87\n')
            for path in paths:
                file.write(f'{SCN_TIME}LINE {path["name"]} {path["start_lat"]},{path["start_lon"]} {path["end_lat"]},{path["end_lon"]}\n')
            file.write(f'{SCN_TIME}PAN 37, 14.24\n')
            file.write(f'{SCN_TIME}ZOOM 0.3\n')
            file.write(f'{SCN_TIME}TRAILS ON\n')
            file.write(f'{SCN_TIME}VNAV OFF\n')
            
            existing_locations = [] 

            # Create planes and destinations
            while True:
                try:
                    for path in paths:
                        plane_name = next(plane_names)
                        lat, lon = generate_random_lat_long_within_percentage(path, 0.5, existing_locations)  # Spawn within first 50% of the path
                        existing_locations.append((lat, lon)) 
                        # time_str = time.strftime("%H:%M:%S.%f")[:-3]
                        random_speed = random.randint(175, 250)
                        file.write(f'{SCN_TIME}CRE {plane_name}, B738, {lat},{lon}, 90, 25000, {random_speed}\n')
                        file.write(f'{SCN_TIME}DEST {plane_name}, {path["end_lat"]},{path["end_lon"]}\n')
                    # time += time_delta  # Increment time after each set of planes
                except StopIteration:
                    break  # Stop when all plane names are used

            file.write(f'{SCN_TIME}PLUGIN BravoPlugin\n')


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Create scenario files for Bravo Environment.")
    parser.add_argument("-s", type=int, help="Number of scenarios to be generated", required=True)
    parser.add_argument("-t", type=str, help="Type of scenario to be generated (TRAIN, TEST, VAL)", required=True)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate scenarios
    generate_scenarios(args.s, args.t)
