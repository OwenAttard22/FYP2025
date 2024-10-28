""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, net, sim, traf  #, settings, navdb, traf, sim, scr, tools
import numpy as np
from gym import Env, spaces

myclientrte = None

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    global env
    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'MLCONTROL',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 1.0,

        'update':          update,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
        }

    stackfunctions = {
        # The command name for your function
        'MLSTEP': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'MLSTEP',

            # A list of the argument types your function accepts. For a description of this, see ...
            '',

            # The name of your function in this plugin
            mlstep,

            # a longer help text of your function.
            'Simulate one MLCONTROL time interval.']
    }
    
    env = BlueSkyEnv()

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    global env
    
    data = dict(
        lat=traf.lat,
        lon=traf.lon,
        alt=traf.alt
    )
    # net.send_event(b'MLSTATEREPLY', data, myclientrte)
    env.run(data)
    sim.hold()

def preupdate():
    pass

def reset():
    global env
    pass

def mlstep():
    global myclientrte
    myclientrte = stack.routetosender()
    sim.op()
    
class BlueSkyEnv(Env):
    def __init__(self):
        super(BlueSkyEnv, self).__init__()
        
        # Define the action and observation space for Gymnasium
        # For simplicity, let's assume we are controlling the aircraft's heading (1 action)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space includes latitude, longitude, and altitude of all aircraft
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(traf.lat), 3), dtype=np.float32)

        # Internal simulation state
        self.state = None
        self.current_step = 0
        self.max_steps = 500  # Number of steps before resetting
        
    def reset(self):
        """ Resets the environment to its initial state and returns initial observation"""
        sim.reset()
        self.current_step = 0
        
        # Initialize BlueSky with random aircraft data or a specific scenario
        aclat = np.random.uniform(51, 54, len(traf.lat))  # Random latitudes
        aclon = np.random.uniform(2, 8, len(traf.lon))   # Random longitudes
        achdg = np.random.randint(1, 360, len(traf.lat))  # Random headings
        acalt = np.random.uniform(1000, 35000, len(traf.alt))  # Random altitudes
        
        # Create aircraft in the BlueSky simulation
        traf.create(aclat=aclat, aclon=aclon, achdg=achdg, acspd=np.ones(len(traf.lat)) * 250, acalt=acalt)
        
        # Return the initial observation (state)
        self.state = np.column_stack((traf.lat, traf.lon, traf.alt))
        return self.state
    
    def step(self, action):
        """Executes the given action in the environment and returns the next state, reward, done, and info."""
        self.current_step += 1
        
        # Apply action to BlueSky (e.g., modify aircraft headings)
        # Assuming action is the change in heading for the first aircraft in the environment
        new_heading = (traf.hdg[0] + action[0] * 15) % 360  # Example: Action scales heading change
        traf.ap.selhdgcmd(0, new_heading)  # Apply the action to the first aircraft

        sim.step()  # Step the simulation

        # Get the new state
        self.state = np.column_stack((traf.lat, traf.lon, traf.alt))

        # Compute reward (for example, proximity to a goal, avoiding collisions)
        reward = self._compute_reward()

        # Check if the episode is done (e.g., max steps reached, goal achieved)
        done = self.current_step >= self.max_steps

        # Additional info (if any)
        info = {}

        return self.state, reward, done, info

    def render(self, mode='human'):
        """Renders the environment. Not implemented for BlueSky as it's a headless simulator."""
        pass

    def close(self):
        """Cleanup the environment when done."""
        sim.reset()  # Reset the simulation when closing

    def run(self, data):
        """Pass simulation state to the environment."""
        self.state = np.column_stack((data['lat'], data['lon'], data['alt']))

    def _compute_reward(self):
        """Custom reward calculation based on the current state."""
        # Example reward: Distance from a target, or minimizing collisions
        # Currently, reward is zero by default
        return 0.0