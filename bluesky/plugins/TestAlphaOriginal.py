import threading
import pygame
import random
from bluesky import core, stack, traf, tools
import time
import geopy.distance
import math
import sys
import signal, os
import socket, json

HOST = "127.0.0.1"
PORT = 8000

LMML_LAT, LMML_LONG = 35.8575, 14.4775  # Malta International Airport
LICJ_LAT, LICJ_LONG = 38.1864, 13.0914  # Palermo Airport
LICC_LAT, LICC_LONG = 37.4667, 15.0664  # Catania Airport

DESTINATIONS = {
    "HAWK": [LMML_LAT, LMML_LONG],
    "SCOPRION": [LMML_LAT, LMML_LONG],
    "PHOENIX": [LMML_LAT, LMML_LONG],
    "PEGASUS": [LMML_LAT, LMML_LONG],
    "WRAITH": [LMML_LAT, LMML_LONG],
    "THUNDERBIRD": [LMML_LAT, LMML_LONG],
    "EAGLE": [LICJ_LAT, LICJ_LONG],
    "VIPER": [LICJ_LAT, LICJ_LONG],
    "SPARROW": [LICJ_LAT, LICJ_LONG],
    "TALON": [LICJ_LAT, LICJ_LONG],
    "LIGHTING": [LICJ_LAT, LICJ_LONG],
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

    # Instantiate the plugin class and start PyGame in a separate thread
    plugin_instance = TestAlpha()
    pygame_thread = threading.Thread(target=plugin_instance.run)
    pygame_thread.start()  # Start PyGame in a new thread

    return config

class Plane:
    def __init__(self, name, lat, long, speed, altitude, heading):
        self.name = name
        self.lat = lat
        self.long = long
        self.speed = speed
        self.altitude = altitude
        self.heading = heading
        self.dist_to_waypoint = 0.0  # Initialize with default value
        self.qdr_to_waypoint = 0.0   # Initialize with default value
        
        self.neighbours = [None,None,None,None,None,None] # format of neighbours: Neighbour1, Dist1, Brg1, Neighbour2, Dist2, Brg2

    def update_position(self, lat, long, heading, dist_to_waypoint, qdr_to_waypoint):
        """ Update the plane's position and waypoint information. """
        self.lat = lat
        self.long = long
        self.heading = heading
        self.dist_to_waypoint = dist_to_waypoint
        self.qdr_to_waypoint = qdr_to_waypoint
        
    def update_neighbours(self, neighbour1, dist1, bearing1, neighbour2, dist2, bearing2):
        self.neighbours = [neighbour1, dist1, bearing1, neighbour2, dist2, bearing2]

class TestAlpha(core.Entity):
    ''' A plugin that retrieves and updates aircraft information using Plane objects and renders them with PyGame. '''

    def __init__(self):
        super().__init__()
        # os.environ["SDL_VIDEODRIVER"] = "dummy"  # hide pygame window
        
        # Initialize the plane dictionary
        self.planes = {}
        self.init_planes()
        # PyGame setup
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 800))
        pygame.display.set_caption("Aircraft Information Display")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
                
                # print(f"Received data: {data}")
                split_data = data.split("\n")
                split_data = [msg.strip() for msg in split_data if msg.strip()] # remove empty string
                for split in split_data:
                    message = json.loads(split)

                    if "reset" in message and message["reset"] is True:
                        print("Received reset command from Gym. Resetting environment...")
                        self.action_reset()

                    # Process actions
                    if "actions" in message:
                        self.action_apply_action(message["actions"])

                    # Process planes that are marked as done
                    if "done_planes" in message:
                        for plane_id in message["done_planes"]:
                            print(f"❌ To remove {plane_id}")
                            self.action_plane_done(plane_id)

                    if "type" in message and message["type"] == "observations":

                        observations = {}
                        for plane_id, plane in self.planes.items():
                            observations[plane_id] = {
                                "lat": plane.lat,
                                "long": plane.long,
                                "heading": plane.heading,
                                "dist_to_wpt": plane.dist_to_waypoint,
                                "qdr_to_wpt": plane.qdr_to_waypoint,
                                "neighbour1_dist": plane.neighbours[1],
                                "neighbour1_bearing": plane.neighbours[2],
                                "neighbour2_dist": plane.neighbours[4],
                                "neighbour2_bearing": plane.neighbours[5],
                            }

                        # print(f"Sending Observations: {observations.keys()}")
                        conn.sendall(json.dumps(observations).encode())

            except Exception as e:
                print(f"Error in TCP communication: {e}")
        
    def init_planes(self):
        for plane_id in traf.id:
            name = traf.id[traf.id2idx(plane_id)]
            lat = traf.lat[traf.id2idx(plane_id)]
            lon = traf.lon[traf.id2idx(plane_id)]
            speed = traf.tas[traf.id2idx(plane_id)]
            altitude = traf.alt[traf.id2idx(plane_id)]
            heading = traf.hdg[traf.id2idx(plane_id)]
            self.planes[plane_id] = Plane(name, lat, lon, speed, altitude, heading)

    def update(self):
        """Update each Plane object's latitude, longitude, heading, distance to waypoint, and QDR to waypoint."""
        
        landed_planes = [plane_id for plane_id in self.planes if plane_id not in traf.id]
        for plane_id in landed_planes:
            # Send plane done message
            self.planes.pop(plane_id)
            
        # if len(self.planes) <= 2:
        #     print("Not enough planes, terminating...")
        #     stack.stack("QUIT")
        #     self.running = False
        #     return
        
        # Update plane positions as before
        for plane_id in traf.id:
            plane = self.planes.get(plane_id)
            if plane:
                lat = traf.lat[traf.id2idx(plane_id)]
                lon = traf.lon[traf.id2idx(plane_id)]
                hdg = traf.hdg[traf.id2idx(plane_id)]
                # print(plane, plane_id)
                call_sign = plane_id.upper()
                destination = DESTINATIONS.get(call_sign, [0, 0])
                _, qdr_to_wpt = tools.geo.qdrdist(lat, lon, destination[0], destination[1])
                dist_to_wpt = geopy.distance.distance((lat, lon), (destination[0], destination[1])).km
                if dist_to_wpt > 600:
                    print(plane_id, dist_to_wpt)
                plane.update_position(lat, lon, hdg, dist_to_wpt, qdr_to_wpt)

        # Check for new echo messages
        self.check_for_echo_messages()
        
        self.find_neighbours()

        time.sleep(0.1)

    def check_for_echo_messages(self):
        """Method to retrieve and process any echo messages from BlueSky."""
        echo_messages = getattr(traf, 'echo', [])  # Adjust as necessary if echo is stored elsewhere in traf
        if echo_messages:
            for message in echo_messages:
                print(f"Received ECHO: {message}")
                # Process each message as needed, e.g., display in PyGame or log it.
    
    # def render(self):
    #     """Render aircraft as moving objects on a map with fixed airports and neighbor connections."""
        
    #     # Define map boundaries (Modify if needed)
    #     MAP_TOP_LEFT = (40.0, 7.0)  # (Max Lat, Min Lon)
    #     MAP_BOTTOM_RIGHT = (33.0, 22.0)  # (Min Lat, Max Lon)
        
    #     SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 800  # Adjust if needed
    #     self.screen.fill((30, 30, 30))  # Dark background

    #     # Function to convert lat/lon into screen coordinates
    #     def to_screen_coords(lat, lon):
    #         x = int((lon - MAP_TOP_LEFT[1]) / (MAP_BOTTOM_RIGHT[1] - MAP_TOP_LEFT[1]) * SCREEN_WIDTH)
    #         y = int((MAP_TOP_LEFT[0] - lat) / (MAP_TOP_LEFT[0] - MAP_BOTTOM_RIGHT[0]) * SCREEN_HEIGHT)
    #         return x, y

    #     # Draw Fixed Airports
    #     airports = {
    #         "LMML": (LMML_LAT, LMML_LONG),
    #         "LICJ": (LICJ_LAT, LICJ_LONG),
    #         "LICC": (LICC_LAT, LICC_LONG)
    #     }
        
    #     for name, (lat, lon) in airports.items():
    #         x, y = to_screen_coords(lat, lon)
    #         pygame.draw.circle(self.screen, (255, 255, 0), (x, y), 10)  # Yellow airport dot
    #         text_surface = self.font.render(name, True, (255, 255, 0))
    #         self.screen.blit(text_surface, (x + 5, y - 20))

    #     # Store plane positions for rendering neighbor connections
    #     plane_positions = {}

    #     # First pass: Draw all planes and store their positions
    #     for plane in self.planes.values():
    #         x, y = to_screen_coords(plane.lat, plane.long)
    #         plane_positions[plane.name] = (x, y)
    #         self.draw_plane(x, y, plane)

    #     # Second pass: Draw all neighbour connections
    #     for plane in self.planes.values():
    #         if plane.neighbours:
    #             for i in range(0, 6, 3):
    #                 neighbor_name = plane.neighbours[i]
    #                 dist = plane.neighbours[i + 1]
    #                 if neighbor_name and dist:
    #                     if neighbor_name in plane_positions:
    #                         neighbor_x, neighbor_y = plane_positions[neighbor_name]

    #                         # line colour based on distance
    #                         if dist > 150:
    #                             color = (65, 169, 204)  # Blue
    #                         elif 101 <= dist <= 150:
    #                             color = (135, 194, 83)  # Green
    #                         elif 51 <= dist < 101:
    #                             color = (194, 148, 83)  # Orange
    #                         else:
    #                             color = (194, 83, 83)  # Red

    #                         pygame.draw.line(self.screen, color, plane_positions[plane.name], (neighbor_x, neighbor_y), 2)

    #                         # Display distance value near the line
    #                         mid_x, mid_y = (plane_positions[plane.name][0] + neighbor_x) // 2, (plane_positions[plane.name][1] + neighbor_y) // 2
    #                         text = self.font.render(f"{dist:.1f} km", True, color)
    #                         self.screen.blit(text, (mid_x, mid_y))

    #     pygame.display.flip()

        
    def find_neighbours(self):
        '''Computes the two nearest neighbours for each plane'''
        
        positions = {plane_id: (plane.lat, plane.long) for plane_id, plane in self.planes.items()}
               
        for plane_id, (lat, long) in positions.items():
            distances = []
            
            for other_id, (other_lat, other_lon) in positions.items():
                if plane_id != other_id:
                    distance = geopy.distance.distance((lat, long), (other_lat, other_lon)).km
                    _, bearing = tools.geo.qdrdist(lat, long, other_lat, other_lon)
                    distances.append((other_id, distance, bearing))
                    
            distances.sort(key=lambda x: x[1])  # sort distances list
        
            if len(distances) >= 2:
                neighbour1, dist1, bearing1 = distances[0]
                neighbour2, dist2, bearing2 = distances[1]
            elif len(distances) == 1:
                neighbour1, dist1 = distances[0]
                neighbour2, dist2 = None, None
            else:
                neighbour1, dist1, neighbour2, dist2 = None, None, None, None
                
            if plane_id in self.planes:
                self.planes[plane_id].update_neighbours(neighbour1, dist1, bearing1, neighbour2, dist2, bearing2)
                # print(f"ID: {plane_id}, {neighbour1}, {dist1}, {bearing1}, {neighbour2}, {dist2}, {bearing2}")
    
    
    # def draw_plane(self, x, y, plane):
    #     """Draws a small square representing the plane and displays its info."""
    #     size = 10  # square size
        
    #     pygame.draw.rect(self.screen, (0, 255, 0), (x - size//2, y - size//2, size, size))

    #     # Display plane info (name, heading, distance)
    #     plane_text = f"{plane.name} {plane.heading:.1f}° {plane.dist_to_waypoint:.1f}km"
    #     text_surface = self.font.render(plane_text, True, (255, 255, 255))
    #     self.screen.blit(text_surface, (x + 8, y - 10))  

    def action_apply_action(self, actions):
        '''Applies heading changes as found in actions list of format [plane_id, heading]'''
        for plane_id, action in actions.items():
            if plane_id in self.planes:
                traf.hdg[traf.id2idx(plane_id)] = action
                # AUTOPILOT ISSUE
                
    def action_reset(self):
        '''Resets the environment and planes to their initial state'''
        self.planes.clear()
        stack.stack('RESET')
        time.sleep(0.1)
        #open alpha/train/train_0002
        stack.stack(f"OPEN alpha/train/train_0001")
        self.init_planes()

    def action_plane_done(self, plane_id):
        if plane_id in self.planes:
            print("Attempting to delete plane, ", plane_id)
            stack.stack(f"DEL {plane_id}")
            del self.planes[plane_id]

    def run(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.update()
            # self.render()
            self.clock.tick(20)

        pygame.quit()
        # os.kill(os.getpid(), signal.SIGTERM)
        
