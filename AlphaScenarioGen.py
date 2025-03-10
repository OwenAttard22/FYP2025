import os
import re
import tqdm
import random
import argparse
import math

# Directory paths
TRAIN = "./scenario/Alpha/TRAIN"
TEST = "./scenario/Alpha/TEST"
VAL = "./scenario/Alpha/VAL"

COOL_PLANE_NAMES = [
    'HAWK', 'EAGLE', 'FALCON', 'SCORPION', 'VIPER',
    'RAVEN', 'PHOENIX', 'SPARROW', 'HORNET', 'PEGASUS',
    'TALON', 'GRYPHON', 'WRAITH', 'LIGHTNING', 'DRAGON',
    'THUNDERBIRD', 'STORM', 'BLADE'
]

LMML_LAT = 35.857012
LMML_LONG = 14.477101
LMML_WPT_LAT = 35.727825
LMML_WPT_LONG = 14.628304

LICJ_LAT = 38.1759987
LICJ_LONG = 13.0909996
LICJ_WPT_LAT = 38.22035
LICJ_WPT_LONG = 13.08989

LICC_LAT = 37.4668007
LICC_LONG = 15.0663996
LICC_WPT_LAT = 37.468164
LICC_WPT_LONG = 15.11516

CIRCLE_RADIUS = 20

AREA_MAX_LAT = 40.06
AREA_MIN_LAT = 33.62
AREA_MAX_LONG = 21.87
AREA_MIN_LONG = 7.97

SCN_TIME = "00:00:00.00>"

NAUTICAL_MILE = 1852
MIN_DISTANCE_NM = 10
R = 6371000  # Radius of the Earth in meters

# Ensure directories exist
os.makedirs(TRAIN, exist_ok=True)
os.makedirs(TEST, exist_ok=True)
os.makedirs(VAL, exist_ok=True)

def haversine(lat1, lon1, lat2, lon2): # Calculate the great-circle distance between two points (lat1, lon1) and (lat2, lon2) in meters.
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c  # Distance in meters

def generate_random_lat_long(existing_locations):
    def random_lat_long():
        return random.uniform(AREA_MIN_LAT, AREA_MAX_LAT), random.uniform(AREA_MIN_LONG, AREA_MAX_LONG)
    
    while True:
        lat, lon = random_lat_long()
        is_valid = True
        
        for existing_lat, existing_lon in existing_locations:
            distance = haversine(lat, lon, existing_lat, existing_lon) / NAUTICAL_MILE  # Convert meters to nautical miles
            if distance < MIN_DISTANCE_NM:
                is_valid = False
                break
        
        if is_valid:
            return lat, lon

def get_next_index(directory, scenario_type):
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
    # Map type to its respective directory
    directories = {"TRAIN": TRAIN, "TEST": TEST, "VAL": VAL}
    directory = directories.get(scenario_type.upper())
    
    if not directory:
        raise ValueError(f"Invalid scenario type: {scenario_type}. Use TRAIN, TEST, or VAL.")
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Determine starting index
    start_index = get_next_index(directory, scenario_type)
    
    # Airport data (cyclic distribution)
    airports = [
        {"name": "LMML", "lat": LMML_LAT, "long": LMML_LONG, "wpt_lat": LMML_WPT_LAT, "wpt_long": LMML_WPT_LONG},
        {"name": "LICJ", "lat": LICJ_LAT, "long": LICJ_LONG, "wpt_lat": LICJ_WPT_LAT, "wpt_long": LICJ_WPT_LONG},
        {"name": "LICC", "lat": LICC_LAT, "long": LICC_LONG, "wpt_lat": LICC_WPT_LAT, "wpt_long": LICC_WPT_LONG},
    ]
    
    for i in tqdm.tqdm(range(scenarios)):
        unique_id = f"{scenario_type}_{start_index + i:04d}"
        
        # Define the file path
        file_path = os.path.join(directory, f"{unique_id}.scn")
        
        existing_locations = []  # To track generated locations and ensure minimum distance
        
        # Write content to the file
        with open(file_path, "w") as file:
            file.write(f'{SCN_TIME}CRECMD COLOUR 255,255,255\n')
            file.write(f'{SCN_TIME}AREA 40.06,7.97 33.62,21.87\n')
            file.write(f'{SCN_TIME}CIRCLE LMML_CIRCLE, {LMML_LAT},{LMML_LONG}, {CIRCLE_RADIUS}\n')
            file.write(f'{SCN_TIME}CIRCLE LICJ_CIRCLE, {LICJ_LAT},{LICJ_LONG}, {CIRCLE_RADIUS}\n')
            file.write(f'{SCN_TIME}CIRCLE LICC_CIRCLE, {LICC_LAT},{LICC_LONG}, {CIRCLE_RADIUS}\n')
            file.write(f'{SCN_TIME}PAN 37, 14.24\n')
            file.write(f'{SCN_TIME}ZOOM 0.3\n')
            file.write(f'{SCN_TIME}TRAILS ON\n')
            
            for idx, plane_name in enumerate(COOL_PLANE_NAMES):  # Plane-specific commands
                lat, lon = generate_random_lat_long(existing_locations)
                existing_locations.append((lat, lon))  # Store generated location
                
                # Distribute planes across airports
                airport = airports[idx % len(airports)]  # Cycle through airports
                
                file.write(f'{SCN_TIME}CRE {plane_name}, B747, {lat},{lon}, 50, 10000, 300\n')
                file.write(f'{SCN_TIME}ADDWPT {plane_name}, {airport["wpt_lat"]},{airport["wpt_long"]}\n')
                file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO DELRTE {plane_name}\n')
                file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO ECHO {plane_name} ON APPROACH\n')
                file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO DEST {plane_name}, {airport["name"]}\n')
                file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO SPD {plane_name} 50\n')
                file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO COLOUR {plane_name} 255,68,51\n')
                file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO ALT {plane_name} 1000\n')
                file.write(f'{SCN_TIME}{plane_name} ATDIST {airport["name"]} 20 SPD {plane_name} 100\n')
                file.write(f'{SCN_TIME}{plane_name} ATDIST {airport["name"]} 20 COLOUR {plane_name} 249,166,2\n')
                file.write(f'{SCN_TIME}{plane_name} ATDIST {airport["name"]} 20 ALT {plane_name} 5000\n')
                file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO {plane_name} AT {airport["name"]} DO ECHO {plane_name} LANDED\n')
                file.write(f'{SCN_TIME}{plane_name} AT {plane_name}001 DO {plane_name} AT {airport["name"]} DO DEL {plane_name}\n')
            
            file.write(f'{SCN_TIME}OP\n')
            file.write(f'{SCN_TIME}FF\n')

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Create scneario files for Alpha Environment.")
    
    # Add arguments
    parser.add_argument("-s", type=str, help="Number of scenarios to be generated")
    parser.add_argument("-t", type=str, help="Type of scenario to be generated (TRAIN, TEST, VAL)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate scenarios
    generate_scenarios(int(args.s), args.t)