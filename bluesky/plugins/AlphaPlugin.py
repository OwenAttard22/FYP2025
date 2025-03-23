import threading
import random
from bluesky import core, stack, traf, tools
import time
import geopy.distance
from geopy.distance import geodesic
import math
import socket, json
from haversine import haversine, Unit
import sys
# from bluesky import bs

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
    # server_thread = threading.Thread(target=plugin_instance.manage_connection)
    # server_thread.start()
    run_thread = threading.Thread(target=plugin_instance.run, daemon=True)
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
        self.socket = None
        
        self.connection = threading.Thread(target=self.manage_connection, daemon=True)
        self.connection.start()
        
        # self.server_thread = threading.Thread(target=self.start_server)
        # self.server_thread.daemon = True
        # self.server_thread.start()

    def manage_connection(self):
        """Handles connecting to Gym Environment and processing data."""
        retry_count = 1
        max_retries = 5

        while self.running:
            try:
                print(f"🔄 Connecting to environment at {HOST}:{PORT}... ({retry_count}/{max_retries})")
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((HOST, PORT))
                print("✅ Connected to environment!")

                while self.running:
                    try:
                        buffer = self.socket.recv(4096).decode()
                        if not buffer:
                            print("⚠️ No data received. Closing connection.")
                            break  

                        messages = buffer.strip().split("\n")
                        for msg in messages:
                            try:
                                message = json.loads(msg)

                                if message.get("reset"):
                                    print("🔄 Received reset command. Stopping thread...")
                                    self.action_reset()

                                elif "actions" in message:
                                    self.action_apply_action(message["actions"])

                                elif "done_planes" in message:
                                    for plane_id in message["done_planes"]:
                                        print(f"❌ Removing {plane_id}")
                                        self.action_plane_done(plane_id)

                                elif message.get("type") == "observations":
                                    self.send_observations()

                            except json.JSONDecodeError:
                                print("⚠️ Invalid JSON received. Skipping...")

                    except socket.error as e:
                        print(f"⚠️ Connection error while receiving data: {e}")
                        break  

                retry_count = 1  

            except socket.error as e:
                print(f"⚠️ Connection failed: {e}. Retrying in 3 seconds...")
                retry_count += 1
                if retry_count > max_retries:
                    print("🛑 Maximum retries reached. Stopping plugin.")
                    self.running = False
                    stack.stack("QUIT")
                time.sleep(3)

        print("🔄 Exiting connection thread.")
            
    def send_observations(self):
        """ Sends current aircraft observations to Gym """
        try:
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
            # print(observations)
            message = json.dumps(observations) + "\n"
            self.socket.sendall(message.encode())
        except socket.error:
            print("⚠️ Connection error while sending observations. Reconnecting...")
            self.manage_connection()


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
    
    def action_plane_done(self, plane_id):
        if plane_id in self.planes:
            print("Attempting to delete plane, ", plane_id)
            stack.stack(f"DEL {plane_id}")
            del self.planes[plane_id]
        
    def action_reset(self):
        """Gracefully stop all threads and exit the program."""
        stack.stack(f"RESET")
        print("Bluesky reset")
        # Send QUIT command to BlueSky
        stack.stack(f"QUIT")
        print("Bluesky quit")

        # Stop all running loops
        self.running = False  

        # Close socket to force exit from recv()
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except Exception as e:
                print(f"⚠️ Error closing socket: {e}")

        # List active threads before joining
        print(f"⚠️ Active threads before join: {threading.enumerate()}")

        # Ensure manage_connection exits cleanly **without joining itself**
        if threading.current_thread() is not self.connection and hasattr(self, "connection"):
            if self.connection.is_alive():
                print("🔴 Stopping manage_connection thread...")
                self.connection.join(timeout=3)

        # Ensure run_thread exits cleanly
        if hasattr(self, "run_thread") and threading.current_thread() is not self.run_thread:
            if self.run_thread.is_alive():
                print("🔴 Stopping run thread...")
                self.run_thread.join(timeout=3)

        # List active threads after joining
        print(f"⚠️ Active threads after join: {threading.enumerate()}")

        time.sleep(1)

        print("🛑 Exiting program...")

        # Try normal exit
        sys.exit(0)

        # If sys.exit() fails, force terminate the process
        os._exit(1)

        
        # try:
        #     status = {"status": "closed"}
        #     message = json.dumps(status) + "\n"
        #     self.socket.sendall(message.encode())
        #     sys.exit(0)
        # except socket.error:
        #     print("⚠️ Connection error while sending closed status message.")
        
    # def action_reset(self):
    #     self.planes.clear()
    #     stack.stack('RESET')
    #     time.sleep(0.5)
        
    #     stack.stack("OPEN alpha/train/train_0011")
    #     time.sleep(2)
    #     # self.manage_connection()
        
    #     # self.init_planes()
        
    #     self.running = False

    def run(self):
        while self.running:
            self.update()
            # time.sleep(0.1)
        
    def stop(self):
        """Gracefully stop the plugin and cleanup resources."""
        print("Stopping plugin...")

        # Close the socket server
        try:
            self.socket.close()
        except Exception as e:
            print(f"Error closing socket: {e}")

        print("Plugin stopped successfully.")