import gym
import numpy as np
from gym import spaces
import socket
import json
import pygame
import time
from Selector import Select

DISTANCES = {}

# Constants
HOST = "127.0.0.1"
PORT = 8000
MAX_AGENTS = 4 # Adjust based on scenario
# CALLSIGNS = ['HAWK', 'EAGLE', 'FALCON', 'SCORPION', 'VIPER', 'RAVEN', 'PHOENIX', 'SPARROW', 'HORNET', 'PEGASUS', 'TALON', 'GRYPHON', 'WRAITH', 'LIGHTNING', 'DRAGON', 'THUNDERBIRD', 'STORM', 'BLADE']

class AlphaEnv(gym.Env):
    """ Multi-Agent Gymnasium Environment for BlueSky ATC """

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_agents=4, action_space_type="discrete"):
        super(AlphaEnv, self).__init__()
        
        self.num_agents = num_agents
        self.action_space_type = action_space_type
        self.running = False
        
        self.done_n = [False] * self.num_agents
        self.removed_n = [False] * self.num_agents # to track if planes have been removed from scenario once landed
        self.reward_n = [0] * self.num_agents
        
        # Normalisation Constants
        self.lat_max = 40.06
        self.lat_min = 33.62
        self.lon_max = 21.87
        self.lon_min = 7.97
        self.dist_max = 1500
        self.dist_min = 0
        
        # Start TCP Server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(1)
        print("Started TCP Server on Port 8000")

        # Define action space: Discrete or Continuous
        if action_space_type == "discrete":
            self.action_space = [spaces.Discrete(9) for _ in range(num_agents)]  # [-20, -15, -10, -5, 0, 5, 10, 15, 20]
        else:
            self.action_space = [spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32) for _ in range(num_agents)] 

        # Define observation space
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32) for _ in range(num_agents)
        ]
    
    def reset(self):
        """ Reset the environment and receive the initial state """
        print("Reset function called")
        
        if not self.running:
            print("Not running, starting BlueSky Plugin...")
        
            Select()
            time.sleep(3)
            
            print("Waiting for BlueSky Plugin to connect...")
            self.conn, self.addr = self.server_socket.accept()
            print(f"Connected to BlueSky Plugin at {self.addr}")
            
            self.running = True
        
        else:
            print("Environment already running, resetting...")
        
            if hasattr(self, 'screen'):
                pygame.display.quit()
                pygame.quit()
            
            self.done_n = [False] * self.num_agents  # Reset done flags
            self.reward_n = [0] * self.num_agents  # Reset reward list
            self.removed_n = [False] * self.num_agents
            
            # Reset state
            self.send_action({"reset": True})
            print("Reset action sent")
            time.sleep(5)
            
            Select()
            time.sleep(5)
            
            # Close and re-open connection to plugin
            self.conn.close()
            time.sleep(5)
            self.conn, self.addr = self.server_socket.accept()
            print(f"Reconnected to BlueSky Plugin at {self.addr}")
            
            self.running = True
            
        time.sleep(2)
        
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 800))
        pygame.display.set_caption("BlueSky ATC - Multi-Agent Environment")
        self.font = pygame.font.Font(None, 24)
        
        self.send_action({"type": "observations"})
        return self.receive_observations()
        # return

    def step(self, action_n=None):
        """ Execute a step in the environment with multiple agent actions """
        self.reward_n = [0] * self.num_agents  # reset reward list
        self.send_action({"type": "observations"})
        
        # Receive new observations
        observations, formatted_obs = self.receive_observations()
        # print(formatted_obs)
        callsigns = list(observations.keys())
        # print("Callsigns: ", callsigns)
        
        # # Send action to BlueSky
        # action_data = json.dumps({"actions": action_n})
        # self.client_socket.sendall(action_data.encode())

        self.reward_n, self.done_n = self.compute_rewards(observations)
        
        # done_planes = [callsigns[i] for i in range(self.num_agents) if self.done_n[i]]
        
        done_planes = []
        for i in range(self.num_agents):
            # print("i: ", i)
            if self.done_n[i] and not self.removed_n[i]:
                done_planes.append(callsigns[i])
                self.removed_n[i] = True
                
        if done_planes:
            # print("Done Planes: ", done_planes)
            self.send_action({"done_planes": done_planes})
        
        return observations, formatted_obs, self.reward_n, self.done_n, {}

    def receive_observations(self):
        """ Requests and Receives aircraft data from the BlueSky plugin """

        try:

            # Set a timeout to prevent blocking indefinitely
            # self.client_socket.settimeout(3.0)
            self.server_socket.settimeout(3.0)

            buffer = ""
            while True:
                # chunk = self.client_socket.recv(4096).decode()  # Receive chunk of data
                # chunk = self.server_socket.recv(4096).decode()  # Receive chunk of data
                chunk = self.conn.recv(4096).decode()

                if not chunk:
                    break  # Stop if no more data

                buffer += chunk

                try:
                    observations = json.loads(buffer)
                    # print("Observations received", observations)
                    formatted = [self.format_observation(obs) for obs in observations.values()]
                    # print(formatted)
                    return observations, formatted
                except json.JSONDecodeError:
                    # print("Incomplete JSON received, waiting for more data...")
                    continue  # Wait for more data to complete JSON

        except socket.timeout:
            print("Timeout: No data received from BlueSky within 3 seconds.")
            return {}, [np.zeros(9) for _ in range(self.num_agents)]  # Return empty data

        except Exception as e:
            print(f"Connection Error: {e}")
            return {}, [np.zeros(9) for _ in range(self.num_agents)]  # Return empty data



    def format_observation(self, obs):
        """ Convert observation JSON into numpy array """
        return np.array([
            ((obs["lat"] - self.lat_min)/(self.lat_max - self.lat_min)),  # normalise latitude
            ((obs["long"] - self.lon_min)/(self.lon_max - self.lon_min)),  # normalise longitude
            ((obs["heading"] - 0)/(360 - 0)),  # normalise heading 
            ((obs["dist_to_wpt"] - self.dist_min)/(self.dist_max - self.dist_min)), # normalise distance
            ((obs["qdr_to_wpt"] - 0)/(360 - 0)),  # normalise bearing
            ((obs["neighbour1_dist"] - self.dist_min)/(self.dist_max - self.dist_min)),
            ((obs["neighbour1_bearing"] - 0)/(360 - 0)),
            ((obs["neighbour2_dist"] - self.dist_min)/(self.dist_max - self.dist_min)),
            ((obs["neighbour2_bearing"] - 0)/(360 - 0))
        ], dtype=np.float32)

    def compute_rewards(self, observations):
        """ Compute rewards based on the current state following rc, ra, rt, rr 
            - r_c = Penalty for being too close to another aircraft (<10 km)
            - r_a = Penalty for deviating too far from destination
            - r_t = Small penalty per timestep to encourage faster arrival
            - r_r = Reward for reaching the destination
        """
        
        global DISTANCES
        
        comp_rewards_n = [0] * self.num_agents

        for i, obs in enumerate(observations):
            if obs is None or self.done_n[i]:  # Skip processing if plane is already done
                continue
            
            lat, lon, heading, dist_to_wpt, qdr_to_wpt, n1_dist, n1_bearing, n2_dist, n2_bearing = observations[obs].values()
            if obs == 'FALCON':
                if obs not in DISTANCES:
                    DISTANCES[obs] = dist_to_wpt
                else:
                    DISTANCES[obs] = min(DISTANCES[obs], dist_to_wpt)

            if n1_dist < 10:
                rc = -1
                self.done_n[i] = True
            elif n2_dist < 10:
                rc = -1
                self.done_n[i] = True
            else:
                rc = 0
                
            ra = -0.5 if dist_to_wpt > 500 else 0 # subject to change I.E. PLEASE CHANGE
            rt = -0.01  
            rr = 1 if dist_to_wpt < 1 else 0  

            comp_rewards_n[i] = rc + ra + rt + rr
            
            # if dist_to_wpt < 10:
            #     print(obs, dist_to_wpt)
            
            self.done_n[i] = True if (dist_to_wpt < 5) else False

        # print(DISTANCES)
        return comp_rewards_n, self.done_n

    def send_action(self, action):
        """ Sends action data to BlueSky """
        action_data = json.dumps(action) + "\n"
        # self.client_socket.sendall(action_data.encode())
        self.conn.sendall(action_data.encode())
        

    def render(self, observations, mode='human'):
        """ Render aircraft positions and relevant information on a 2D Pygame map """

        if not hasattr(self, 'screen'):  # Initialize Pygame only if not already done
            pygame.init()
            self.screen = pygame.display.set_mode((1000, 800))  # Set screen size
            pygame.display.set_caption("BlueSky ATC - Multi-Agent Environment")
            self.font = pygame.font.Font(None, 24)

        self.screen.fill((30, 30, 30))  # Dark background

        # Define Fixed Airports (Reference Points)
        airports = {
            "LMML": (35.8575, 14.4775),  # Malta
            "LICJ": (38.1864, 13.0914),  # Palermo
            "LICC": (37.4667, 15.0664)   # Catania
        }

        # Convert latitude/longitude to screen coordinates
        def to_screen_coords(lat, lon):
            """ Convert real-world coordinates to screen space """
            lat = float(lat)
            lon = float(lon)
            MAP_TOP_LEFT = (40.06, 7.97)  # (Max Lat, Min Lon)
            MAP_BOTTOM_RIGHT = (33.62, 21.87)  # (Min Lat, Max Lon)
            SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 800  # Match Pygame screen size
            x = int((lon - MAP_TOP_LEFT[1]) / (MAP_BOTTOM_RIGHT[1] - MAP_TOP_LEFT[1]) * SCREEN_WIDTH)
            y = int((MAP_TOP_LEFT[0] - lat) / (MAP_TOP_LEFT[0] - MAP_BOTTOM_RIGHT[0]) * SCREEN_HEIGHT)
            return x, y

        # Draw Airport Locations
        for name, (lat, lon) in airports.items():
            x, y = to_screen_coords(lat, lon)
            pygame.draw.circle(self.screen, (255, 255, 0), (x, y), 10)  # Yellow for Airports
            text_surface = self.font.render(name, True, (255, 255, 0))
            self.screen.blit(text_surface, (x + 5, y - 20))
        # print("Render Observation", observations)

        # Draw All Aircraft
        for i, data in enumerate(observations):
            # print("DATA", data, observations[data]["lat"], observations[data]["long"])
            x, y = to_screen_coords(observations[data]["lat"], observations[data]["long"])  # (lat, long)

            # Determine color based on distance to waypoint
            dist_to_wpt = observations[data]["dist_to_wpt"]
            if dist_to_wpt > 150:
                color = (65, 169, 204)  # Blue (Far)
            elif 101 <= dist_to_wpt <= 150:
                color = (135, 194, 83)  # Green (Mid-Range)
            elif 51 <= dist_to_wpt < 101:
                color = (194, 148, 83)  # Orange (Approaching)
            else:
                color = (194, 83, 83)  # Red (Very Close)

            # Draw aircraft as small squares
            pygame.draw.rect(self.screen, color, (x - 5, y - 5, 10, 10))

            # Display Aircraft Information (Callsign, Heading, Distance to Waypoint)
            plane_text = (
                f"Plane {i+1} | {observations[data]["heading"]:.1f}°\n"
                f"Dist: {observations[data]["dist_to_wpt"]:.1f} km | Bearing: {observations[data]["qdr_to_wpt"]:.1f}°"
            )
            text_lines = plane_text.split("\n")

            for j, line in enumerate(text_lines):
                text_surface = self.font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (x + 12, y + (j * 15)))  # Offset text to the right

        pygame.display.flip()
        
    def dummy(self, observations):
        return [4] * self.num_agents
    
    def close(self):
        """ Close the environment and terminate the connection """
        
        if hasattr(self, 'screen'):
                pygame.display.quit()
                pygame.quit()
            
        self.done_n = [False] * self.num_agents  # Reset done flags
        self.reward_n = [0] * self.num_agents  # Reset reward list
        self.removed_n = [False] * self.num_agents
        
        # Reset state
        self.send_action({"reset": True})
        print("Reset action sent")
        time.sleep(5)
        
        # Close and re-open connection to plugin
        self.conn.close()
        
        self.running = False
        self.server_socket.close()
        print("Environment closed")


def run():
    env = AlphaEnv()
    
    while True:
        # print("Resetting environment...")
        # obs = env.reset()
        # print("Reset done")
        running  = True
        
        while running:
            print("⚠️ Starting new episode...")
            obs, reward_n, done_n, _ = env.step()
            env.render(obs)
            active_planes = sum(not done for done in done_n)
            if active_planes <= 3:
                print("Epsidoe done, resetting...")
                env.reset()
                print("Reset done")
                # running = False
                
        print("⚠️ ERORR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                
    # env.close()
    

def test():
    env = AlphaEnv()
    
    test_plane = "LIGHTNING"
    action_index = 0  # Track action changes
    discrete_actions = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

    while True:
        print(f"⏳ Waiting 7 seconds before sending new action...")
        time.sleep(7)

        # Select an action from the predefined list
        action_value = discrete_actions[action_index]
        action_index = (action_index + 1) % len(discrete_actions)

        print(f"🚀 Sending heading change {action_value}° to plane {test_plane}")

        env.send_action({"actions": {test_plane: action_value}})

        obs, reward_n, done_n, _ = env.step()

        env.render(obs)
    
if __name__ == "__main__":
    # test()
    run()