from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from CentralisedCritic import CCPPOPolicy, CentralisedLSTMModel
from gymnasium import spaces
import numpy as np
from AlphaEnv2 import AlphaEnv
from ray.tune.registry import register_env
from Callbacks import MyCallbacks
import os

# Register custom model
ModelCatalog.register_custom_model("My_CC_LSTM_PPO", CentralisedLSTMModel)

observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
action_space = spaces.Discrete(9)

# Register AlphaEnv
register_env("AlphaEnv", lambda config: AlphaEnv(config))

# Checkpoint
checkpoint_path = os.path.abspath("./checkpoints/Alpha/")
os.makedirs(checkpoint_path, exist_ok=True)

config = (
    PPOConfig()
    .environment(env="AlphaEnv", env_config={})
    .framework("torch")
    .env_runners(num_env_runners=0)
    .callbacks(MyCallbacks)
    .training(
        gamma=0.99,
        lr=1e-4,
        clip_param=0.2,
        train_batch_size=300,
        # batch_mode="complete_episodes",
        model={
            "custom_model": "My_CC_LSTM_PPO",
            "max_seq_len": 20,
            "use_lstm": True, 
            "lstm_use_prev_action_reward": True,
            "opponent_action_dim": 3,
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
    .resources(num_gpus=1)
    .experimental(_validate_config=False)
)

def main():
    config.batch_mode = "complete_episodes"
    trainer = config.build()

    for i in range(5):  # Start with 5 quick iterations
        print(f"🔁 Iteration {i}")
        result = trainer.train()
        print(result)
        print("-" * 40)
        
        # save checkpoint
        checkpoint_path = trainer.save_to_path(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        print("-" * 40)
        
config.batch_mode = "complete_episodes"
trainer = config.build()
trainer.restore_from_path(checkpoint_path)
print("Checkpoint restored from", checkpoint_path)

for i in range(3):
    print(f"🔁 Iteration {i}")
    result = trainer.train()
    print(result)
    print("-" * 40)
    
    # save checkpoint
    checkpoint_path = trainer.save_to_path(checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    print("-" * 40)
    
