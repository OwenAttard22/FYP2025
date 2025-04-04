from AlphaEnv2 import AlphaEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib import marl

class AlphaEnvMARLlib(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        self.env = AlphaEnv(env_config)

    def reset(self, *, seed=None, options=None):
        return self.env.reset()

    def step(self, action_dict):
        return self.env.step(action_dict)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

env = marl.make_env(environment_name="custom", env_class=AlphaEnvMARLlib)