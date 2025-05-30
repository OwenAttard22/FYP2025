from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.registry import POLICIES
from ray.rllib.models import ModelCatalog
from CentralisedCritic import CCPPOPolicy, CentralisedCriticModel
from gymnasium import spaces
import numpy as np
# from AlphaEnv2 import AlphaEnv
from AlphaEnv1 import AlphaEnv
from BravoEnv import BravoEnv
from CharlieEnv import CharlieEnv
from ray.tune.registry import register_env
from Callbacks import MyCallbacks, MyCallbacks2
import os
import sys

# Register custom model
ModelCatalog.register_custom_model("CentralisedCriticModel", CentralisedCriticModel)
POLICIES["CCPPOPolicy"] = "CentralisedCritic"

observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
action_space = spaces.Discrete(7) # action space config
# action_space = spaces.Box(low=np.array([-10.0]), high=np.array([10.0]), dtype=np.float32)

# Register AlphaEnv
register_env("AlphaEnv", lambda config: AlphaEnv(config))
register_env("BravoEnv", lambda config: BravoEnv(config))
register_env("CharlieEnv", lambda config: CharlieEnv(config))

# Checkpoint

# checkpoint_path = os.path.abspath("./checkpoints/Training/Alpa/AlphaEnv1/Train_ExperimentF3")
# restore_path = os.path.abspath("./checkpoints/Training/Alpa/AlphaEnv1/Train_ExperimentF3")

# checkpoint_path = os.path.abspath("./checkpoints/Training/Alpa/AlphaEnv2/Model1")
# restore_path = os.path.abspath("./checkpoints/Training/Alpa/AlphaEnv2/Model1")

# checkpoint_path = os.path.abspath("./checkpoints/Training/BravoEnv/Discrete7")
# restore_path = os.path.abspath("./checkpoints/Training/BravoEnv/Discrete7")

# checkpoint_path = os.path.abspath("./checkpoints/Training/BravoEnv/Discrete5")
# checkpoint_path = os.path.abspath("./checkpoints/Training/BravoEnv/Discrete5_100")
# restore_path = os.path.abspath("./checkpoints/Training/BravoEnv/Discrete5_100")

# checkpoint_path = os.path.abspath("./checkpoints/Training/BravoEnv/Discrete5_100_2")
# restore_path = os.path.abspath("./checkpoints/Training/BravoEnv/Discrete5_100_2")


# checkpoint_path = os.path.abspath("./checkpoints/Training/BravoEnv/Continuous")
# restore_path = os.path.abspath("./checkpoints/Training/BravoEnv/Continuous")

# checkpoint_path = os.path.abspath("./checkpoints/Training/CharlieEnv/Discrete")
# restore_path = os.path.abspath("./checkpoints/Training/CharlieEnv/Discrete")

checkpoint_path = os.path.abspath("./checkpoints/Training/CharlieEnv/Continuous")
restore_path = os.path.abspath("./checkpoints/Training/CharlieEnv/Continuous")

os.makedirs(checkpoint_path, exist_ok=True)

# Form Tune AlphaEnv1
# clip param: 0.379451
# lr: 0.000422
# gamma 0.996607
AlphaEnv1 = {
    "name": "AlphaEnv",
    "clip_param": 0.379451,
    "lr": 0.000422,
    "gamma": 0.996607,
}

BravoEnv1 = {
    "name": "BravoEnv",
    "lr": 0.000089,
    "gamma": 0.902796,
    "clip_param": 0.340814,
}

CharlieEnv1 = {
    "name": "CharlieEnv",
    "lr": 0.000047,
    "gamma": 0.973648,
    "clip_param": 0.284006,
}

params = CharlieEnv1

config = (
    PPOConfig()
    .environment(env=params["name"], env_config={})
    .framework("torch")
    .env_runners(num_env_runners=0)
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .callbacks(MyCallbacks2)
    .training(
        gamma=params["gamma"],
        lr=params["lr"],
        clip_param=params["clip_param"],
        train_batch_size=20000,
        model={
                "custom_model": "CentralisedCriticModel",
                "custom_model_config": {
                    "num_agents": 18,
                    "action_size": 1,
                }
            }
    )
    .multi_agent(
        policies={
            "shared_policy": (
                CCPPOPolicy,
                observation_space,
                action_space,
                {},
            )
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    )
    .resources(num_cpus_for_main_process=15)
    .reporting(keep_per_episode_custom_metrics=True)
    .experimental(_validate_config=True)
)

if "--restore" in sys.argv:
    config.batch_mode = "complete_episodes"
    trainer = config.build_algo()
    trainer.load_checkpoint(restore_path)
    print("Checkpoint restored from", restore_path)

    for i in range(100):
        print(f"🔁 Iteration {i}")
        result = trainer.train()
        print(result)
        print("-" * 40)
        
        if "--test" not in sys.argv:
            # save checkpoint
            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            print("-" * 40)
else:
    config.batch_mode = "complete_episodes"
    trainer = config.build_algo()

    for i in range(100):
        print(f"🔁 Iteration {i}")
        result = trainer.train()
        print(result)
        print("-" * 40)
        
        # save checkpoint
        trainer.save_checkpoint(checkpoint_path)
        
        print(f"Checkpoint saved at {checkpoint_path}")
        print("-" * 40)
    
