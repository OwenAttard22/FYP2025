import numpy as np
from gymnasium import spaces
import socket
import json
import pygame
import time
from Selector import Select

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Constants
HOST = "127.0.0.1"
PORT = 8000
# MAX_AGENTS = 4 # Adjust based on scenario
# CALLSIGNS = ['HAWK', 'EAGLE', 'FALCON', 'SCORPION', 'VIPER', 'RAVEN', 'PHOENIX', 'SPARROW', 'HORNET', 'PEGASUS', 'TALON', 'GRYPHON', 'WRAITH', 'LIGHTNING', 'DRAGON', 'THUNDERBIRD', 'STORM', 'BLADE']

class AlphaEnv(MultiAgentEnv):
    """ Multi-Agent Environment for BlueSky ATC """

    def __init__(self, config=None):
        super().__init__()
        
        self.agents = self.possible_agents = ["EAGLE", "FALCON", "SCORPION", "HAWK"]
        self.running = False
        
        self.done_dict = {agent_id: False for agent_id in self.agents}  # Done flags for each agent
        self.removed_dict = {agent_id: False for agent_id in self.agents} # to track if planes have been removed from scenario once landed
        self.reward_dict = {agent_id: 0 for agent_id in self.agents}
        
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
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(1)
        print("Started TCP Server on Port 8000")

        self.observation_spaces = {
            agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
            for agent_id in self.agents
        }
          
        self.action_spaces = {
            agent_id: spaces.Discrete(9)
            for agent_id in self.agents
        }
    
    def reset(self, *, seed=None, options=None):
        """ Reset the environment and receive the initial state """
        print("Reset function called")
        
        if not self.running:
            print("Not running, starting BlueSky Plugin...")
            
            print("Called new scenario")
            Select()
            time.sleep(5)
            
            try:
                print("🔄 Waiting for BlueSky to connect on port 8000...")
                self.conn, self.addr = self.server_socket.accept()
                print(f"✅ Accepted connection from {self.addr}")
            except socket.timeout:
                print("❌ Timeout waiting for BlueSky to connect.")
                raise
            print(f"Connected to BlueSky Plugin at {self.addr}")
            
            self.running = True
        
        else:
            print("Environment already running, resetting...")
        
            if hasattr(self, 'screen'):
                pygame.display.quit()
                pygame.quit()
            
            self.done_dict = {agent_id: False for agent_id in self.agents}  # Reset done flags
            self.reward_dict = {agent_id: 0 for agent_id in self.agents}
            self.removed_dict = {agent_id: False for agent_id in self.agents}
            
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
        
        empty_info_dict = {agent_id: {} for agent_id in self.agents} # empty info dict
        return self.receive_observations(), empty_info_dict

    def step(self, action_dict):
        """ Execute a step in the environment with multiple agent actions """
        
        action_dict = {agent: int(action) for agent, action in action_dict.items()}
        
        self.reward_dict = {agent_id: 0 for agent_id in self.agents} # reset reward list
        
        for agent, action in action_dict.items():
            print("Agent: ", agent, type(agent))
            print("Action: ", action, type(action))
    
        self.send_action({"actions": action_dict})
            
        
        self.send_action({"type": "observations"})
        time.sleep(0.1)
        
        # Receive new observations
        observations = self.receive_observations()

        self.reward_dict, self.done_dict = self.compute_rewards(observations)
        
        
        new_done_planes = []
        for agent_id in self.agents:
            if self.done_dict[agent_id] and not self.removed_dict[agent_id]:
                new_done_planes.append(agent_id)
                self.removed_dict[agent_id] = True
                
        if new_done_planes:
            self.send_action({"done_planes": new_done_planes})
            
        self.done_dict["__all__"] = sum(self.done_dict[agent] for agent in self.agents) >= len(self.agents) - 3
            
        truncated_dict = {agent_id: False for agent_id in self.agents} # empty
        truncated_dict["__all__"] = False
        
        info_dict = {agent_id: {} for agent_id in self.agents} # empty
        
        return observations, self.reward_dict, self.done_dict, truncated_dict, info_dict

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
                    formatted = {callsign: self.format_observation(obs) for callsign, obs in observations.items()}
                    # print(formatted)
                    return formatted
                except json.JSONDecodeError:
                    # print("Incomplete JSON received, waiting for more data...")
                    continue  # Wait for more data to complete JSON

        except socket.timeout:
            print("Timeout: No data received from BlueSky within 3 seconds.")
            return {agent_id: np.zeros(9) for agent_id in self.agents} # return empty observation

        except Exception as e:
            print(f"Connection Error: {e}")
            return {agent_id: np.zeros(9) for agent_id in self.agents} # return empty observation

    def normalise(self, value, type):
        """ Normalise values based on type """
        if type == "lat":
            return (value - self.lat_min) / (self.lat_max - self.lat_min)
        elif type == "long":
            return (value - self.lon_min) / (self.lon_max - self.lon_min)
        elif type == "dist":
            return (value - self.dist_min) / (self.dist_max - self.dist_min)
        elif type == "hdg":
            return (value - 0) / (360 - 0)
        else:
            raise ValueError("Unknown normalisation type")

    def format_observation(self, obs):
        """ Convert observation JSON into numpy array """
        return np.array([
            (self.normalise(obs["lat"], "lat")),
            (self.normalise(obs["long"], "long")),
            (self.normalise(obs["heading"], "hdg")),
            (self.normalise(obs["dist_to_wpt"], "dist")),
            (self.normalise(obs["qdr_to_wpt"], "hdg")),
            (self.normalise(obs["neighbour1_dist"], "dist")),
            (self.normalise(obs["neighbour1_bearing"], "hdg")),
            (self.normalise(obs["neighbour2_dist"], "dist")),
            (self.normalise(obs["neighbour2_bearing"], "hdg"))
        ], dtype=np.float32)
        
    def _outside_airspace(self, lat, lon):
        """ Check if the aircraft is outside the airspace """
        return not (0 <= lat <= 1 and 0 <= lon <= 1) 
    
    def compute_rewards(self, observations):
        """ Compute rewards based on the current state following rc, ra, rt, rr 
            - r_c = Penalty for being too close to another aircraft (<10 km)
            - r_a1 = Penalty for deviating too far from destination (>700km)
            - r_a2 = Penalty for being too far from destination, exiting airspace
            - r_t = Small penalty per timestep to encourage faster arrival
            - r_r = Reward for reaching the destination
        """
        
        # Constants
        rc_dist = self.normalise(10, "dist")  # 10 km
        ra1_dist = self.normalise(700, "dist")  # 700 km
        rr_dist = self.normalise(2, "dist")  # 2 km
        
        global DISTANCES
        
        temp_rewards_dict = {agent_id: 0 for agent_id in self.agents}
        
        for agent, obs in observations.items():
            print("Agent: ", agent)
            print("Observation: ", obs)
            
            if self.done_dict[agent] or obs is None:
                continue
            
            lat, lon, heading, dist_to_wpt, qdr_to_wpt, n1_dist, n1_bearing, n2_dist, n2_bearing = obs
            
            if n1_dist < rc_dist and n2_dist < rc_dist:
                r_c = -2
                self.done_dict[agent] = True
            elif n1_dist < rc_dist:
                r_c = -1
                self.done_dict[agent] = True
            elif n2_dist < rc_dist:
                r_c = -1
                self.done_dict[agent] = True
            else:
                r_c = 0
                
            r_a1 = -0.05 if dist_to_wpt > ra1_dist else 0
            r_t = -0.005
            
            if dist_to_wpt <= rr_dist:
                r_r = 1
                print("✅", agent, "landed!")
            else:
                r_r = 0
                
            if self._outside_airspace(lat, lon):
                r_a2 = -1
                self.done_dict[agent] = True
            else:
                r_a2 = 0
            
            temp_rewards_dict[agent] = r_c + r_a1 + r_t + r_r + r_a2
            
            self.done_dict[agent] = True if (dist_to_wpt < rr_dist) else False
            
        return temp_rewards_dict, self.done_dict

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
    
    def close(self):
        """ Close the environment and terminate the connection """
        
        if hasattr(self, 'screen'):
                pygame.display.quit()
                pygame.quit()
        
        # # Reset state
        # self.send_action({"reset": True})
        # print("Reset action sent")
        # time.sleep(5)
        
        # Close and re-open connection to plugin
        self.conn.close()
        
        self.running = False
        self.server_socket.close()
        print("Environment closed")

def run():
    env = AlphaEnv()
    obs, _ = env.reset()
    done = {"__all__": False}
    step_count = 0

    while not done["__all__"]:
        # Sample a random action for each active agent
        action_dict = {
            agent: env.action_spaces[agent].sample()
            for agent in obs
        }

        # Step the environment
        obs, rewards, dones, truncs, infos = env.step(action_dict)

        # Print results
        print(f"\nStep {step_count}")
        print("Actions:", action_dict)
        print("Observations:", obs)
        print("Rewards:", rewards)
        print("Dones:", dones)
        print("Truncs:", truncs)

        # Optional: render the environment
        # env.render(obs)

        done = dones
        step_count += 1

    print("✅ Episode finished.")
    env.close()
    
if __name__ == "__main__":
    # test()
    run()
    
    
    
