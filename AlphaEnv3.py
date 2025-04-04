import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from AlphaEnv2 import AlphaEnv
import numpy as np
from gymnasium import spaces
import tensorflow as tf

# Shared policy function
def policy_mapping_fn(agent_id, episode, **kwargs):
    return "shared_policy"


if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    # ✅ Register the env BEFORE config.build()
    register_env("AlphaEnv", lambda config: AlphaEnv(config))

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
    action_space = spaces.Discrete(9)

    config = (
        PPOConfig()
        .environment(env="AlphaEnv", env_config={})
        .framework("tf")
        .env_runners(num_env_runners=0)
        .training(
            gamma=0.99,
            lr=1e-4,
            train_batch_size=10000,
            clip_param=0.2
        )
        .multi_agent(
            policies={
                "shared_policy": (
                    None,  # use default policy class
                    observation_space,
                    action_space,
                    {}
                )
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .resources(num_gpus=0)
        .experimental(_validate_config=False)
    )

    # Build and train
    ppo = config.build()
    for i in range(1):
        print(f"🔁 Training iteration {i}")
        result = ppo.train()
        print(result)
        # checkpoint_dir = ppo.save_to_path()
        print("-"*20)
        # print(f"Checkpoint saved in {checkpoint_dir}")