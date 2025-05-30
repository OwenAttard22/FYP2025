from ray.rllib.algorithms.callbacks import DefaultCallbacks

class MyCallbacks2(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Grab your real env (unwrapped) from the base_env
        # env = base_env.get_unwrapped()[0]
        
        env = base_env.envs[0].unwrapped
        
        collided = env.collided
        landed   = env.landed
        outsided = env.outsided

        # Push them into RLlib’s custom_metrics so they show up in the result dict:
        episode.custom_metrics["collided"] = collided
        episode.custom_metrics["landed"]   = landed
        episode.custom_metrics["outsided"] = outsided
        episode.hist_data["collided"] = [collided]
        episode.hist_data["landed"] = [landed]
        episode.hist_data["outsided"] = [outsided]
        print(f"Episode ended with collided: {collided}, landed: {landed}, outsided: {outsided}")
        # print(episode)



from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

class MyCallbacks(RLlibCallback):        
    def on_episode_end(
        self,
        *,
        episode: MultiAgentEpisode,
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
        no_change = env.envs[0].unwrapped.no_change
        
        metrics_logger.log_value(key="collided", value=collided, window=1)
        metrics_logger.log_value(key="landed", value=landed, window=1)
        metrics_logger.log_value(key="outsided", value=outsided, window=1)
        metrics_logger.log_value(key="no_change", value=no_change, window=1)
        # metrics_logger.log_value("test_collided", collided, reduce="sum", window=None, clear_on_reduce=True)
        # metrics_logger.log_value("test_collided2", collided, reduce="mean", window=None, clear_on_reduce=True)
        # metrics_logger.log_value("test_collide3", collided, reduce="sum", window=1000, clear_on_reduce=True)
        # metrics_logger.log_value("test_collided4", collided, reduce="mean", window=1000, clear_on_reduce=True)
        # metrics_logger.log_value("test_collided5", collided, reduce="sum", window=None, clear_on_reduce=False)
        # metrics_logger.log_value("test_collided6", collided, reduce="mean", window=None, clear_on_reduce=False)
        # metrics_logger.log_value("test_collide7", collided, reduce="sum", window=1000, clear_on_reduce=False)
        # metrics_logger.log_value("test_collided8", collided, reduce="mean", window=1000, clear_on_reduce=False)
        
        episode.custom_metrics["collided"] = collided
        episode.custom_metrics["landed"] = landed
        episode.custom_metrics["outsided"] = outsided
        episode.custom_metrics["no_change"] = no_change
        
    # def on_train_result(
    #     self,
    #     *,
    #     algorithm,
    #     metrics_logger,
    #     result,
    #     **kwargs
    # ):
    #     # Print the training result
    #     print("Training result:", result)