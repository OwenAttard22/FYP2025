import threading
import random
from bluesky import core, stack, traf, tools
import time
import geopy.distance
from geopy.distance import geodesic
import math
import socket, json
from haversine import haversine, Unit

HOST = "127.0.0.1"
PORT = 8000

LMML_LAT, LMML_LONG = 35.857, 14.477  # Malta International Airport
LICJ_LAT, LICJ_LONG = 38.186, 13.091  # Palermo Airport
LICC_LAT, LICC_LONG = 37.466, 15.066  # Catania Airport

DESTINATIONS = {
    "HAWK": [LMML_LAT, LMML_LONG],
    "SCORPION": [LMML_LAT, LMML_LONG],
    "PHOENIX": [LMML_LAT, LMML_LONG],
    "PEGASUS": [LMML_LAT, LMML_LONG],
    "WRAITH": [LMML_LAT, LMML_LONG],
    "THUNDERBIRD": [LMML_LAT, LMML_LONG],
    "EAGLE": [LICJ_LAT, LICJ_LONG],
    "VIPER": [LICJ_LAT, LICJ_LONG],
    "SPARROW": [LICJ_LAT, LICJ_LONG],
    "TALON": [LICJ_LAT, LICJ_LONG],
    "LIGHTNING": [LICJ_LAT, LICJ_LONG],
    "STORM": [LICJ_LAT, LICJ_LONG],
    "FALCON": [LICC_LAT, LICC_LONG],
    "RAVEN": [LICC_LAT, LICC_LONG],
    "HORNET": [LICC_LAT, LICC_LONG],
    "GRYPHON": [LICC_LAT, LICC_LONG],
    "DRAGON": [LICC_LAT, LICC_LONG],
    "BLADE": [LICC_LAT, LICC_LONG],
}

def init_plugin():
    """ Initialization function for the test plugin. """
    config = {
        'plugin_name': 'TestAlpha2',
        'plugin_type': 'sim'
    }

    plugin_instance = TestAlpha()
    server_thread = threading.Thread(target=plugin_instance.start_server)
    server_thread.start()
    run_thread = threading.Thread(target=plugin_instance.run)
    run_thread.start()

    return config

class Plane:
    def __init__(self, name, lat, long, speed, altitude, heading):
        self.name = name
        self.lat = lat
        self.long = long
        self.speed = speed
        self.altitude = altitude
        self.heading = heading
        self.dist_to_waypoint = 0.0
        self.qdr_to_waypoint = 0.0
        self.neighbours = [None, None, None, None, None, None]

    def update_position(self, lat, long, heading, dist_to_waypoint, qdr_to_waypoint):
        self.lat = lat
        self.long = long
        self.heading = heading
        self.dist_to_waypoint = dist_to_waypoint
        self.qdr_to_waypoint = qdr_to_waypoint

    def update_neighbours(self, neighbour1, dist1, bearing1, neighbour2, dist2, bearing2):
        self.neighbours = [neighbour1, dist1, bearing1, neighbour2, dist2, bearing2]

class TestAlpha(core.Entity):
    def __init__(self):
        super().__init__()
        self.planes = {}
        self.init_planes()
        
        self.running = True
        
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)

        print(f"Server started on {HOST}:{PORT}, waiting for a connection...")

        conn, addr = server_socket.accept()
        print(f"Connected to {addr}")

        while True:
            try:
                data = conn.recv(4096).decode()
                if not data:
                    continue
                
                split_data = data.split("\n")
                split_data = [msg.strip() for msg in split_data if msg.strip()]
                for split in split_data:
                    message = json.loads(split)

                    if "reset" in message and message["reset"] is True:
                        print("Received reset command from Gym. Resetting environment...")
                        self.action_reset()

                    if "actions" in message:
                        self.action_apply_action(message["actions"])

                    if "done_planes" in message:
                        for plane_id in message["done_planes"]:
                            print(f"❌ To remove {plane_id}")
                            self.action_plane_done(plane_id)

                    if "type" in message and message["type"] == "observations":
                        observations = {
                            plane_id: {
                                "lat": plane.lat,
                                "long": plane.long,
                                "heading": plane.heading,
                                "dist_to_wpt": plane.dist_to_waypoint,
                                "qdr_to_wpt": plane.qdr_to_waypoint,
                                "neighbour1_dist": plane.neighbours[1],
                                "neighbour1_bearing": plane.neighbours[2],
                                "neighbour2_dist": plane.neighbours[4],
                                "neighbour2_bearing": plane.neighbours[5],
                            } for plane_id, plane in self.planes.items()
                        }
                        conn.sendall(json.dumps(observations).encode())

            except Exception as e:
                print(f"Error in TCP communication: {e}")

    def init_planes(self):
        for plane_id in traf.id:
            name = traf.id[traf.id2idx(plane_id)]
            lat = float(traf.lat[traf.id2idx(plane_id)])
            lon = float(traf.lon[traf.id2idx(plane_id)])
            speed = traf.tas[traf.id2idx(plane_id)]
            altitude = traf.alt[traf.id2idx(plane_id)]
            heading = traf.hdg[traf.id2idx(plane_id)]
            self.planes[plane_id] = Plane(name, lat, lon, speed, altitude, heading)

    def update(self):
        for plane_id in traf.id:
            plane = self.planes.get(plane_id)
            if plane:
                lat = traf.lat[traf.id2idx(plane_id)]
                lon = traf.lon[traf.id2idx(plane_id)]
                hdg = traf.hdg[traf.id2idx(plane_id)]
                call_sign = plane_id.upper()
                destination = DESTINATIONS.get(call_sign, [0, 0])
                _, qdr_to_wpt = tools.geo.qdrdist(lat, lon, destination[0], destination[1])
                dist_to_wpt = haversine((lat, lon), (destination[0], destination[1]), unit=Unit.KILOMETERS)
                plane.update_position(lat, lon, hdg, dist_to_wpt, qdr_to_wpt)
                
        # self.check_for_echo_messages()
        self.find_neighbours()
        time.sleep(0.1)
        
    # def check_for_echo_messages(self):
    #     """Method to retrieve and process any echo messages from BlueSky."""
    #     echo_messages = getattr(traf, 'echo', [])
    #     if echo_messages:
    #         for message in echo_messages:
    #             print(f"Received ECHO: {message}"
        
    def find_neighbours(self):
        '''Computes the two nearest neighbours for each plane'''
        
        positions = {plane_id: (plane.lat, plane.long) for plane_id, plane in self.planes.items()}
                
        for plane_id, (lat, long) in positions.items():
            distances = []
            
            for other_id, (other_lat, other_lon) in positions.items():
                if plane_id != other_id:
                    distance = geopy.distance.distance((lat, long), (other_lat, other_lon)).km
                    # distance = haversine((lat, long), (other_lat, other_lon))
                    _, bearing = tools.geo.qdrdist(lat, long, other_lat, other_lon)
                    distances.append((other_id, distance, bearing))
                    
            distances.sort(key=lambda x: x[1])  # sort distances list
        
            if len(distances) >= 2:
                neighbour1, dist1, bearing1 = distances[0]
                neighbour2, dist2, bearing2 = distances[1]
            elif len(distances) == 1:
                try:
                    neighbour1, dist1 = distances[0]
                    neighbour2, dist2 = None, None
                except:
                    neighbour1, dist1, neighbour2, dist2 = None, None, None, None
            else:
                neighbour1, dist1, neighbour2, dist2 = None, None, None, None
                
            if plane_id in self.planes:
                self.planes[plane_id].update_neighbours(neighbour1, dist1, bearing1, neighbour2, dist2, bearing2)
                # print(f"ID: {plane_id}, {neighbour1}, {dist1}, {bearing1}, {neighbour2}, {dist2}, {bearing2}")

    def action_apply_action(self, actions):
        for plane_id, action in actions.items():
            if plane_id in self.planes:
                new_heading = (self.planes[plane_id].heading + action) % 360
                stack.stack(f"{plane_id} HDG {new_heading}")
                stack.stack(f"DELAY 00:02:00 {plane_id} LNAV ON")

    # def action_reset(self):
    #     self.planes.clear()
    #     stack.stack('RESET')
    #     time.sleep(0.1)
    #     stack.stack("OPEN alpha/train/train_0011")
    #     self.init_planes()
        
    def action_reset(self):
        self.planes.clear()
        stack.stack('RESET')
        time.sleep(0.1)
        stack.stack("OPEN alpha/train/train_0011")
        time.sleep(2)
        self.init_planes()
        # self.running = False

    def run(self):
        while self.running:
            self.update()
            # time.sleep(0.1)

        self.stop()
        
    def stop(self):
        """Gracefully stop the plugin and cleanup resources."""
        print("Stopping plugin...")

        # Set running flag to False to stop threads
        self.running = False

        # Close the socket server
        try:
            self.server_socket.close()
        except Exception as e:
            print(f"Error closing socket: {e}")

        # Wait for threads to terminate
        if self.server_thread.is_alive():
            self.server_thread.join()
        if self.update_thread.is_alive():
            self.update_thread.join()

        print("Plugin stopped successfully.")