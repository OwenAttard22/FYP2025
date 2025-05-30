from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from CentralisedCritic import CCPPOPolicy, CentralisedCriticModel
from gymnasium import spaces
import numpy as np
from AlphaEnv1 import AlphaEnv
from BravoEnv import BravoEnv
from CharlieEnv import CharlieEnv
from ray.tune.registry import register_env
import os
from Callbacks import MyCallbacks2

# Register model and environment
ModelCatalog.register_custom_model("CentralisedCriticModel", CentralisedCriticModel)
register_env("AlphaEnv", lambda config: AlphaEnv(config))
register_env("BravoEnv", lambda config: BravoEnv(config))
register_env("CharlieEnv", lambda config: CharlieEnv(config))


# Observation and action spaces
observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
action_space = spaces.Discrete(7)
# action_space = spaces.Box(low=np.array([129]), high=np.array([319]), dtype=np.float32)
# action_space = spaces.Box(low=np.array([-10.0]), high=np.array([10.0]), dtype=np.float32)

# Checkpoint path
# restore_path = os.path.abspath("./checkpoints/Training/Alpa/AlphaEnv1/Train_ExperimentF3")
# restore_path = os.path.abspath("./checkpoints/Training/Alpa/AlphaEnv2/Model1")
# restore_path = os.path.abspath("./checkpoints/Training/BravoEnv/Discrete5_100_2")

# restore_path = os.path.abspath("./checkpoints/Training/BravoEnv/Continuous")
restore_path = os.path.abspath("./checkpoints/Training/CharlieEnv/Discrete")

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

# Build config
config = (
    PPOConfig()
    .environment(env=None)
    .framework("torch")
    .env_runners(num_env_runners=0)
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .callbacks(MyCallbacks2)
    .training(
        gamma=params["gamma"],
        lr=params["lr"],
        clip_param= params["clip_param"],
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

config = config.env_runners(create_env_on_local_worker=False)


trainer = config.build_algo()
print("DEBUG:", type(trainer))
trainer.restore(restore_path)
print("DEBUG:", type(trainer))

# env = trainer.workers.local_worker().env
# env = AlphaEnv(config={"cf_type":"TEST"})
# env = BravoEnv(config={"cf_type":"TEST"})
env = CharlieEnv(config={"cf_type":"TEST"})

# print(trainer.workers)
# env = trainer.workers.local_worker().env

for episode in range(100):
    obs, _ = env.reset()
    done = {"__all__": False}
    total_reward = {agent_id: 0 for agent_id in obs}

    while not done["__all__"]:
        actions = {
            agent_id: trainer.compute_single_action(obs[agent_id], policy_id="shared_policy")
            for agent_id in obs if agent_id in env.agents
        }
        obs, rewards, done, _, _ = env.step(actions)

        for agent_id, reward in rewards.items():
            total_reward[agent_id] += reward

    # print(f"✅ Episode {episode + 1} total rewards:")
    # for aid, rew in total_reward.items():
    #     print(f"  {aid}: {rew:.2f}")

# Cleanup
env.close()
