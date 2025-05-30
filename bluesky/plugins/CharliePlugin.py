import threading
import random
from bluesky import core, stack, traf, tools, sim
import time
import geopy.distance
from geopy.distance import geodesic
import math
import socket, json
from haversine import haversine, Unit
import sys
import numpy as np
# from bluesky import bs

HOST = "127.0.0.1"
PORT = 9000

KTS = 0.514444 # Knots to m/s 

P1_LAT, P1_LONG = 38.20965, 20.356376 # Path 1
P2_LAT, P2_LONG = 36.5296, 20.356376 # Path 2
P3_LAT, P3_LONG = 38.548577, 14.92395 # Path 3

# ACTIONS ={
#     0: -20,
#     1: -15,
#     2: -10,
#     3: -5,
#     4: 0,
#     5: 5,
#     6: 10,
#     7: 15,
#     8: 20
# }

ACTIONS ={
    0: -15,
    1: -10,
    2: -5,
    3: 0,
    4: 5,
    5: 10,
    6: 15,
}

# ACTIONS ={
#     0: -10,
#     1: -5,
#     2: 0,
#     3: 5,
#     4: 10,
# }

def init_plugin():
    """ Initialization function for the test plugin. """
    config = {
        'plugin_name': 'CharliePlugin',
        'plugin_type': 'sim'
    }

    plugin_instance = TestAlpha()
    # server_thread = threading.Thread(target=plugin_instance.manage_connection)
    # server_thread.start()
    run_thread = threading.Thread(target=plugin_instance.run, daemon=True)
    run_thread.start()

    return config

class Plane:
    def __init__(self, name, lat, long, speed, altitude, heading, simt, destination):
        self.name = name
        self.lat = lat
        self.long = long
        self.speed = speed
        self.altitude = altitude
        self.heading = heading
        self.dist_to_waypoint = 0.0
        self.qdr_to_waypoint = 0.0
        self.simt = simt
        self.neighbours = [None, None, None, None, None, None]
        self.destination = [destination[0], destination[1]]

    def update_position(self, lat, long, speed, dist_to_waypoint, qdr_to_waypoint, simt):
        self.lat = lat
        self.long = long
        self.dist_to_waypoint = dist_to_waypoint
        self.qdr_to_waypoint = qdr_to_waypoint
        self.simt = simt

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
                print(f"Connecting to environment at {HOST}:{PORT}... ({retry_count}/{max_retries})")
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((HOST, PORT))
                print("Connected to environment!")

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
                                    
                                elif "mlstep" in message:
                                    self.mlstep()
                                    
                                elif "continuous_actions" in message:
                                    self.action_apply_continuous_action(message["continuous_actions"])

                                elif "actions" in message:
                                    self.action_apply_action(message["actions"])

                                elif "done_planes" in message:
                                    # print(message)
                                    # print(buffer)
                                    for plane_id in message["done_planes"]:
                                        print(f"Removing {plane_id}")
                                        self.action_plane_done(plane_id)

                                elif message.get("type") == "observations":
                                    self.send_observations()

                            except json.JSONDecodeError:
                                print("⚠️ Invalid JSON received. Skipping...")

                    except socket.error as e:
                        print(f"Connection error while receiving data: {e}")
                        break  

                retry_count = 1  

            except socket.error as e:
                print(f"Connection failed: {e}. Retrying in 3 seconds...")
                retry_count += 1
                if retry_count > max_retries:
                    print("Maximum retries reached. Stopping plugin.")
                    self.running = False
                    stack.stack("QUIT")
                time.sleep(3)

        print("🔄 Exiting connection thread.")
            
    def send_observations(self):
        """ Sends current aircraft observations to Gym """
        try:
            observations = {
                plane_id: {
                    "lat": float(plane.lat),
                    "long": float(plane.long),
                    "speed": int(plane.speed),
                    "dist_to_wpt": float(plane.dist_to_waypoint),
                    "simt": float(plane.simt),
                    "neighbour1_dist": float(plane.neighbours[1]) if plane.neighbours[1] is not None else None,
                    "neighbour1_bearing": float(plane.neighbours[2]) if plane.neighbours[2] is not None else None,
                    "neighbour2_dist": float(plane.neighbours[4]) if plane.neighbours[4] is not None else None,
                    "neighbour2_bearing": float(plane.neighbours[5]) if plane.neighbours[5] is not None else None,
                } for plane_id, plane in self.planes.items()
            }
            # print(observations)
            message = json.dumps(observations) + "\n"
            self.socket.sendall(message.encode())
        except socket.error:
            print("Connection error while sending observations. Reconnecting...")
            self.manage_connection()


    def init_planes(self):
        for plane_id in traf.id:
            name = traf.id[traf.id2idx(plane_id)]
            lat = float(traf.lat[traf.id2idx(plane_id)])
            lon = float(traf.lon[traf.id2idx(plane_id)])
            speed = round(traf.cas[traf.id2idx(plane_id)] / KTS)
            altitude = traf.alt[traf.id2idx(plane_id)]
            heading = traf.hdg[traf.id2idx(plane_id)]
            destination_str = traf.ap.dest[traf.id2idx(plane_id)]
            lat_str, lon_str = destination_str.split(",")
            destination = [float(lat_str), float(lon_str)]
            simt = sim.simt
            self.planes[plane_id] = Plane(name, lat, lon, speed, altitude, heading, simt, destination)

    def update(self):
        simt = sim.simt
        if simt is None:
            simt = 0.0
        for plane_id in traf.id:
            plane = self.planes.get(plane_id)
            if plane:
                lat = traf.lat[traf.id2idx(plane_id)]
                lon = traf.lon[traf.id2idx(plane_id)]
                hdg = traf.hdg[traf.id2idx(plane_id)]
                speed = round(traf.cas[traf.id2idx(plane_id)] / KTS)
                # print(plane_id, speed)
                call_sign = plane_id.upper()
                destination = plane.destination
                _, qdr_to_wpt = tools.geo.qdrdist(lat, lon, destination[0], destination[1])
                dist_to_wpt = haversine((lat, lon), (destination[0], destination[1]), unit=Unit.KILOMETERS)
                plane.update_position(lat, lon, speed, dist_to_wpt, qdr_to_wpt, simt)
                
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
                    neighbour1, dist1, bearing1 = distances[0]
                    neighbour2, dist2, bearing2 = None, None, None
                except:
                    neighbour1, dist1, bearing1, neighbour2, dist2, bearing2 = None, None, None, None, None, None
            else:
                neighbour1, dist1, bearing1, neighbour2, dist2, bearing2 = None, None, None, None, None, None
                
            if plane_id in self.planes:
                self.planes[plane_id].update_neighbours(neighbour1, dist1, bearing1, neighbour2, dist2, bearing2)
                # print(f"ID: {plane_id}, {neighbour1}, {dist1}, {bearing1}, {neighbour2}, {dist2}, {bearing2}")

    def action_apply_action(self, actions):
        for plane_id, action in actions.items():
            if plane_id in self.planes:
                if plane_id not in traf.id:
                    continue
                
                # new_speed = (self.planes[plane_id].speed + ACTIONS[action])
                new_speed = np.clip(self.planes[plane_id].speed + ACTIONS[action], 129, 319)
                idx = traf.id2idx(plane_id)
                
                if idx == -1 or idx >= len(traf.id):
                    continue
                
                traf.ap.selspdcmd(idx, new_speed)
                # stack.stack(f"{plane_id} SPD {new_speed}")
                self.planes[plane_id].speed = new_speed
                
    def action_apply_continuous_action(self, actions):
        for plane_id, action in actions.items():
            if plane_id in self.planes:
                
                if plane_id not in traf.id:
                    continue
                
                action_value = float(np.squeeze(action))
                # new_speed = (self.planes[plane_id].speed + action_value)
                # new_speed = np.clip(self.planes[plane_id].speed + action_value, 129, 319)
                new_speed = action_value
                idx = traf.id2idx(plane_id)
                
                if idx == -1 or idx >= len(traf.id):
                    continue
                
                traf.ap.selspdcmd(idx, new_speed)
                self.planes[plane_id].speed = new_speed
                # stack.stack(f"{plane_id} SPD {new_speed}")


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
            
    def mlstep(self):
        sim.step(dt_increment=1)
        sim.step(dt_increment=1)

        
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
                print(f"Error closing socket: {e}")

        # List active threads before joining
        print(f"Active threads before join: {threading.enumerate()}")

        # Ensure manage_connection exits cleanly **without joining itself**
        if threading.current_thread() is not self.connection and hasattr(self, "connection"):
            if self.connection.is_alive():
                print("Stopping manage_connection thread...")
                self.connection.join(timeout=3)

        # Ensure run_thread exits cleanly
        if hasattr(self, "run_thread") and threading.current_thread() is not self.run_thread:
            if self.run_thread.is_alive():
                print("Stopping run thread...")
                self.run_thread.join(timeout=3)

        # List active threads after joining
        print(f"Active threads after join: {threading.enumerate()}")

        time.sleep(1)

        print("Exiting program...")

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