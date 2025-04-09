from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

# class MyCallbacks(RLlibCallback):
#     def __init__(self):
#         super().__init__()
    
#     def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
#         print("Training result:", result)
#     def on_episode_end(self, *, episode, metrics_logger, **kwargs):
#         # print(episode.agent_rewards.values())
#         # print(episode.env_states)
#         print(episode._unwrapped_env)``

class MyCallbacks(RLlibCallback):
    def on_episode_step(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs
    ):
        
        collided = env.envs[0].unwrapped.collided
        landed = env.envs[0].unwrapped.landed
        outsided = env.envs[0].unwrapped.outsided
        # print(collided, landed, outsided)
        
    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs
    ):
        collided = env.envs[0].unwrapped.collided
        landed = env.envs[0].unwrapped.landed
        outsided = env.envs[0].unwrapped.outsided
        
        metrics_logger.log_value(key="collided", value=collided, window=100)
        metrics_logger.log_value(key="landed", value=landed, window=100)
        metrics_logger.log_value(key="outsided", value=outsided, window=100)
        
    def on_train_result(
        self,
        *,
        algorithm,
        metrics_logger,
        result,
        **kwargs
    ):
        # Print the training result
        print("Training result:", result)