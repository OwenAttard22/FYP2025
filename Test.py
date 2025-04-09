from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from CentralisedCritic import CCPPOPolicy, CentralisedLSTMModel
from gymnasium import spaces
import numpy as np
from AlphaEnv2 import AlphaEnv
from ray.tune.registry import register_env
from Callbacks import MyCallbacks

ModelCatalog.register_custom_model("My_CC_LSTM_PPO", CentralisedLSTMModel)

observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
action_space = spaces.Discrete(9)

register_env("AlphaEnv", lambda config: AlphaEnv(config))

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

config.batch_mode = "complete_episodes"
trainer = config.build()

for i in range(5):  # Start with 5 quick iterations
    print(f"🔁 Iteration {i}")
    result = trainer.train()
    print(result)
    print("-" * 40)