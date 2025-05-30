import os
import numpy as np
from gymnasium import spaces

from ray import tune
from ray.tune.tuner import Tuner
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air import RunConfig

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from CentralisedCritic import CCPPOPolicy, CentralisedCriticModel
from AlphaEnv1 import AlphaEnv
from BravoEnv import BravoEnv
from CharlieEnv import CharlieEnv
from Callbacks import MyCallbacks

# === ENV + MODEL SETUP ===
observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
# action_space = spaces.Box(low=np.array([-10.0]), high=np.array([10.0]), dtype=np.float32)
action_space = spaces.Discrete(7)

register_env("AlphaEnv1", lambda config: AlphaEnv(config))
register_env("BravoEnv", lambda config: BravoEnv(config))
register_env("CharlieEnv", lambda config: CharlieEnv(config))
ModelCatalog.register_custom_model("My_CC_PPO", CentralisedCriticModel)

# === CHECKPOINT DIR ===
# checkpoint_dir = os.path.abspath("./checkpoints/Tune/AlphaEnv1_2") # discrete
# checkpoint_dir = os.path.abspath("./checkpoints/Tune/AlphaEnv1_3") # 
# checkpoint_dir = os.path.abspath("./checkpoints/Tune/AlphaEnv1_4") # test
# checkpoint_dir = os.path.abspath("./checkpoints/Tune/BravoEnv")
checkpoint_dir = os.path.abspath("./checkpoints/Tune/CharlieEnv")
os.makedirs(checkpoint_dir, exist_ok=True)

# === SEARCH SPACE ===
search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "gamma": tune.uniform(0.9, 0.999),
    "clip_param": tune.uniform(0.1, 0.4),
    "train_batch_size": tune.choice([20000]),
}

# === BASE PPO CONFIG ===
base_config = (
    PPOConfig()
    .environment(env="CharlieEnv", env_config={})
    .framework("torch")
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .env_runners(num_env_runners=0)
    # .callbacks(MyCallbacks)
    .training(
        model={
                "custom_model": "My_CC_PPO",
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
    # .resources(num_gpus=1)
    .resources(num_cpus_for_main_process=15)
    .experimental(_validate_config=True)
)
base_config.batch_mode = "complete_episodes"

# === TRAINING FUNCTION ===
from ray.tune import ExperimentAnalysis
def check_best_config():
    # analysis = ExperimentAnalysis(experiment_checkpoint_path=checkpoint_dir+"/optuna_tune")    
    # df = analysis.dataframe(metric="env_runners/episode_reward_mean", mode="max")
    # print(df[["config/lr", "config/gamma", "config/clip_param", "env_runners/episode_reward_mean"]])

    import os
    base_dir = checkpoint_dir + "/optuna_tune"
    experiments = [f.path for f in os.scandir(base_dir) if f.is_file()]
    for path in experiments:
        try:
            analysis = ExperimentAnalysis(experiment_checkpoint_path=path)    
            df = analysis.dataframe(metric="env_runners/episode_reward_mean", mode="max")
            print(df[["config/lr", "config/gamma", "config/clip_param", "config/train_batch_size","env_runners/episode_reward_mean"]])
            best_cfg = df.iloc[0]
            print("Best config found:\n", best_cfg)
            print("-" * 40)
        except:
            pass

    

# === TUNING FUNCTION ===
def run_tuning():
    print("🚀 Starting hyperparameter tuning...")

    optuna_search = OptunaSearch(metric="env_runners/episode_reward_mean", mode="max")
    algo = ConcurrencyLimiter(optuna_search, max_concurrent=1)

    # tuned_config = base_config.update_from_dict(test_search_space)
    tuned_config = base_config.to_dict()
    tuned_config.update(search_space)

    tuner = Tuner(
        trainable="PPO",
        param_space=tuned_config,
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_reward_mean",
            mode="max",
            search_alg=algo,
            num_samples=10, 
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id[:8]}"
        ),
        run_config=RunConfig(
            name="optuna_tune",
            stop={"training_iteration": 5},
            storage_path=checkpoint_dir,
        ),
    )

    results = tuner.fit()
    print(results)
    best_cfg = results.get_best_result(metric="env_runners/episode_reward_mean", mode="max").config
    # best_cfg = results.get_best_result(metric="episode_reward_mean", mode="max").config
    print("🏆 Best config found:\n", best_cfg)
    return best_cfg

# === ENTRY POINT ===
if __name__ == "__main__":
    import sys
    if "--tune" in sys.argv:
        best_cfg = run_tuning()
        print("✅ Done tuning.")
        # Optional: train_from_best_config(best_cfg)
    else:
        check_best_config()


# AlphaEnv1
# clip param: 0.33167539701800375
# lr: 1.4923152097883673e-05
# gamma 0.978812059090693
# train_batch_size: 10000