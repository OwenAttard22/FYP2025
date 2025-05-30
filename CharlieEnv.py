import numpy as np
from gymnasium import spaces
import socket
import json
# import pygame
import time
import math
from Selector import Select2
import os
import csv

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Constants
HOST = "127.0.0.1"
PORT = 9000
# MAX_AGENTS = 4 # Adjust based on scenario
# CALLSIGNS = ['HAWK', 'EAGLE', 'FALCON', 'SCORPION', 'VIPER', 'RAVEN', 'PHOENIX', 'SPARROW', 'HORNET', 'PEGASUS', 'TALON', 'GRYPHON', 'WRAITH', 'LIGHTNING', 'DRAGON', 'THUNDERBIRD', 'STORM', 'BLADE']

class CharlieEnv(MultiAgentEnv):
    """ Multi-Agent Environment for BlueSky ATC """

    def __init__(self, config=None, max_steps=10000, action_type="discrete"):
        super().__init__()
        self.config = config
        self.action_type = action_type
        self.max_steps = max_steps
        
        # self.agents = self.possible_agents = ["EAGLE", "FALCON", "SCORPION", "HAWK"]
        self.possible_agents = ['HAWK', 'EAGLE', 'FALCON', 'SCORPION', 'VIPER', 'RAVEN', 'PHOENIX', 'SPARROW', 'HORNET', 'PEGASUS', 'TALON', 'GRYPHON', 'WRAITH', 'LIGHTNING', 'DRAGON', 'THUNDERBIRD', 'STORM', 'BLADE']
        self.agents = self.possible_agents.copy()
        self.running = False
        self.agent_cutoff = self.config.get("cutoff", 3)
        self.cf_type = self.config.get("cf_type", "TRAIN")
        self.scen_count = 1
        
        self.done_dict = {agent_id: False for agent_id in self.agents}  # Done flags for each agent
        self.removed_dict = {agent_id: False for agent_id in self.agents} # to track if planes have been removed from scenario once landed
        self.reward_dict = {agent_id: 0 for agent_id in self.agents}
        
        # Counters
        self.landed = 0
        self.collided = 0
        self.outsided = 0
        self.no_change = 0
        self.reward_hist = []
        self.step_count = 0
        
        # Normalisation Constants
        self.lat_max = 40.06
        self.lat_min = 33.62
        self.lon_max = 21.87
        self.lon_min = 7.97
        self.dist_max = 1500
        self.dist_min = 0
        self.simt_max = 50000
        
        # Start TCP Server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(1)
        print("Started TCP Server on Port 9000")

        self.observation_spaces = {
            agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
            for agent_id in self.agents
        }

        if self.action_type == "discrete":
            self.action_spaces = {
                agent_id: spaces.Discrete(7)  # 9 discrete actions
                for agent_id in self.agents
            }
        elif self.action_type == "continuous":
            self.action_spaces = {
                agent_id: spaces.Box(low=np.array([129]), high=np.array([319]), dtype=np.float32)
                for agent_id in self.agents
            }
    
    def reset(self, *, seed=None, options=None):
        """ Reset the environment and receive the initial state """
        print("Reset function called")
        
        if not self.running:
            print("Not running, starting BlueSky Plugin...")
            
            print("Called new scenario")
            Select2(cf_env="Charlie", cf_type=self.cf_type, count=self.scen_count, debug=False)    
            self.scen_count += 1
            time.sleep(5)
            
            try:
                print("🔄 Waiting for BlueSky to connect on port 9000...")
                self.conn, self.addr = self.server_socket.accept()
                print(f"✅ Accepted connection from {self.addr}")
            except socket.timeout:
                print("❌ Timeout waiting for BlueSky to connect.")
                raise
            print(f"Connected to BlueSky Plugin at {self.addr}")
            
            self.running = True
        
        else:
            print("Environment already running, resetting...")
            
            self.agents = self.possible_agents.copy()
            
            self.done_dict = {agent_id: False for agent_id in self.agents}  # Reset done flags
            self.reward_dict = {agent_id: 0 for agent_id in self.agents}
            self.removed_dict = {agent_id: False for agent_id in self.agents}
            
            self.landed = 0
            self.collided = 0
            self.outsided = 0
            self.reward_hist = []
            self.step_count = 0
            self.no_change = 0
            
            # Reset state
            self.send_action({"reset": True})
            print("Reset action sent")
            time.sleep(5)
            
            print("Called new scenario")
            Select2(cf_env="Charlie", cf_type=self.cf_type, count=self.scen_count, debug=False)
            self.scen_count += 1
            time.sleep(5)
            
            # Close and re-open connection to plugin
            self.conn.close()
            time.sleep(5)
            self.conn, self.addr = self.server_socket.accept()
            print(f"Reconnected to BlueSky Plugin at {self.addr}")
            
            self.running = True
            
        time.sleep(2)
        self.send_action({"type": "observations"})
        
        empty_info_dict = {agent_id: {} for agent_id in self.agents} # empty info dict
        return self.receive_observations(), empty_info_dict
    
    def _get_actions_dict(self, action_dict):
        if self.action_type == "discrete":
            action_dict = {agent: int(action) for agent, action in action_dict.items()}
        elif self.action_type == "continuous":
            action_dict = {agent: float(np.squeeze(action)) for agent, action in action_dict.items()}
        return action_dict
    
    def _send_actions(self, action_dict):
        if self.action_type == "discrete":
            self.send_action({"actions": action_dict})
        elif self.action_type == "continuous":
            self.send_action({"continuous_actions": action_dict})
        self.send_action({"mlstep": True})
        
    def _done_planes(self):
        new_done_planes = []        
        for agent_id in self.agents:
            if self.done_dict[agent_id] and not self.removed_dict[agent_id]:
                new_done_planes.append(agent_id)
                self.removed_dict[agent_id] = True
                # self.agents.remove(agent_id)
        return new_done_planes
    
    def _active_agents(self):
        return [agent_id for agent_id in self.agents if not self.removed_dict[agent_id]]
                
    def step(self, action_dict):
        """ Execute a step in the environment with multiple agent actions """
        
        # Filter action dict type
        action_dict = self._get_actions_dict(action_dict)

        # Reset reward dict
        self.reward_dict = {agent_id: 0 for agent_id in self._active_agents()}
        
        # Send actions to plugin
        self._send_actions(action_dict)
        
        # Receive new observations
        self.send_action({"type": "observations"})
        
        # Receive new observations
        observations = self.receive_observations()

        # Get rewards and done flags
        self.reward_dict, self.done_dict = self.compute_rewards(observations, action_dict)
        
        new_done_planes = self._done_planes()
        if new_done_planes:
            self.send_action({"done_planes": new_done_planes})
            
        truncated_dict = {agent_id: False for agent_id in self._active_agents()}
        truncated_dict["__all__"] = False
        
        observations = {agent_id: obs for agent_id, obs in observations.items() if not self.removed_dict[agent_id]}
        info_dict = {agent_id: {} for agent_id in observations}
        
        self.done_dict["__all__"] = len(self._active_agents()) <= self.agent_cutoff
        
        if self.done_dict["__all__"]:
            print("Landed", self.landed, "Collisions", self.collided, "Outsided", self.outsided)
        
        if self.done_dict["__all__"] and self.cf_type == "TEST":
            self.log_test_results()
        
        self.step_count += 1
        
        if self.step_count%1000 == 0:
            print("Step count: ", self.step_count)
        
        if self.step_count >= self.max_steps:
            print("Max steps reached, terminating episode..")
            for agent_id in self._active_agents():
                truncated_dict[agent_id] = True
                self.reward_dict[agent_id] = -100
            truncated_dict["__all__"] = True 
            
        return observations, self.reward_dict, self.done_dict, truncated_dict, info_dict

    def log_test_results(self):
        folder = "Testing"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "charlie1_results.csv")
        # filename = os.path.join(folder, "charlie2_results.csv")

        file_exists = os.path.isfile(filename)
        
        # Prepare the row
        row = {
            "step_count": self.step_count,
            "collisions": self.collided,
            "landings": self.landed,
            "breaches": self.outsided,
        }
        
        # Open file in append mode
        with open(filename, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            
            # Write header only once
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)

    def receive_observations(self):
        """ Requests and Receives aircraft data from the BlueSky plugin """

        try:

            self.server_socket.settimeout(3.0)

            buffer = ""
            while True:
                chunk = self.conn.recv(4096).decode()

                if not chunk:
                    break  # Stop if no more data

                buffer += chunk

                try:
                    observations = json.loads(buffer)
                    formatted = {callsign: self.format_observation(obs) for callsign, obs in observations.items()}
                    return formatted
                except json.JSONDecodeError:
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
            if value is None:
                return 1
            return (value - (-180)) / (180 - (-180))
        elif type == "simt":
            return (value - 0) / (self.simt_max - 0)
        elif type == "ndist":
            if value is None:
                return 1
            return (value - 5) / (50 - 5)
        elif type == "spd":
            return (value - 129) / (319 - 129)
        else:
            raise ValueError("Unknown normalisation type")

    def format_observation(self, obs):
        """ Convert observation JSON into numpy array """
        return np.array([
            (self.normalise(obs["lat"], "lat")),
            (self.normalise(obs["long"], "long")),
            (self.normalise(obs["speed"], "spd")),
            (self.normalise(obs["dist_to_wpt"], "dist")),
            (self.normalise(obs["simt"], "simt")),
            (self.normalise(obs["neighbour1_dist"], "ndist")),
            (self.normalise(obs["neighbour1_bearing"], "hdg")),
            (self.normalise(obs["neighbour2_dist"], "ndist")),
            (self.normalise(obs["neighbour2_bearing"], "hdg"))            
        ], dtype=np.float32)
        
    def _outside_airspace(self, lat, lon):
        """ Check if the aircraft is outside the airspace """
        return not (0 <= lat <= 1 and 0 <= lon <= 1) 
    
    def compute_rewards(self, observations, actions_dict):
        """ Compute rewards based on the current state by aggregating sparse and shaped rewards
            - r_c = Penalty for being too close to another aircraft (<10 km)
            - r_a = Penalty for being too far from destination, exiting airspace
            - r_r = Reward for reaching the destination
            - r_d = Shaped reward for being closer to destination
            - r_t = Shaped reward per timestep
            - r_n = Shaped penalty for facing towards neighbours
            
            > Exit Rewards:
                - r_c = -50 for collision, 0 otherwise
                - r_a = -25 for outside airspace, 0 otherwise
                - r_r = 250 for landing, 0 otherwise
            
            > Shaped Rewards:
                - r_d = scale_rate * (1 - (2 * dist_to_wpt)**0.5) if dist_to_wpt < 0.5 else 0
                - r_f = scale_rate * ((-8 * (brg_to_wpt-0.5)**2) + 1)
                - r_t = growth_rate * (math.exp(simt**2) - 1)
        """
        
        # Constants
        rc_dist = self.normalise(5, "dist")
        rr_dist = self.normalise(5, "dist") 
        
        def get_rt(simt):
            return -1 * (math.exp(simt**2) - 1)

        def get_rd(dist_to_wpt):
            if dist_to_wpt < 0.5:
                return 0.5 * (1 - (2 * dist_to_wpt)**0.5)
            else:
                return 0
            
        def get_rn(n1_dist, n2_dist):
            r_n1 = -((1-n1_dist)**2) if n1_dist <= 1 else 0
            r_n2 = -((1-n2_dist)**2) if n2_dist <= 1 else 0
            return r_n1 + r_n2
        
        global DISTANCES
        
        temp_rewards_dict = {agent_id: 0 for agent_id in self.agents}
        
        for agent, obs in observations.items():
            
            if self.done_dict[agent] or obs is None:
                continue
                        
            lat, lon, speed, dist_to_wpt, simt, n1_dist, n1_bearing, n2_dist, n2_bearing = obs
            
            # Exit REWARDS
            
            # Collided
            if n1_dist < rc_dist or n2_dist < rc_dist:
                r_c = -500
                self.done_dict[agent] = True
                self.collided += 1
                temp_rewards_dict[agent] = r_c
                print(agent, "collided", self.collided)
                continue
                
            # Outsided in case landing fails
            if self._outside_airspace(lat, lon):
                print(agent, "outsided", self.outsided)
                r_a = -1
                self.outsided += 1
                self.done_dict[agent] = True
                temp_rewards_dict[agent] = r_a
                continue
            
            # Landed
            if dist_to_wpt < rr_dist:
                r_r = 250
                self.landed += 1
                self.done_dict[agent] = True
                temp_rewards_dict[agent] = r_r
                # print(agent, "landed", self.landed)
                continue
            
            # SHAPED REWARDS
            shaped = get_rd(dist_to_wpt) + get_rt(simt) + get_rn(n1_dist, n2_dist)
            
            temp_rewards_dict[agent] = shaped
            
        return temp_rewards_dict, self.done_dict

    def send_action(self, action):
        """ Sends action data to BlueSky """
        action_data = json.dumps(action) + "\n"
        self.conn.sendall(action_data.encode())
    
    def close(self):
        """ Close the environment and terminate the connection """
    
        self.send_action({"reset": True})
        print("Close action sent")
        self.conn.close()
        
        self.running = False
        self.server_socket.close()
        print("Environment closed")

def run():
    env = CharlieEnv({})
    
    for i in range(5):
        obs, _ = env.reset()
        
        time.sleep(5)
        
        done = {"__all__": False}
        step_count = 0

        while not done["__all__"]:
            # Sample a random action for each active agent
            action_dict = {
                agent: env.action_spaces[agent].sample()
                # agent: 0
                for agent in obs
            }

            # Step the environment
            obs, rewards, dones, truncs, infos = env.step(action_dict)

            # Print results
            # print(f"\nStep {step_count}")
            # print("Actions:", action_dict)
            # print("Observations:", obs)
            # print("Rewards:", rewards)
            # print("Dones:", dones)
            # print("Truncs:", truncs)

            # Optional: render the environment
            # env.render(obs)
            
            if truncs["__all__"]:
                dones["__all__"] = True

            done = dones
            step_count += 1

        print("✅ Episode finished.")
    env.close()
    
if __name__ == "__main__":
    # test()
    run()
    
    
    