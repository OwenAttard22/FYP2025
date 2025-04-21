import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.framework import try_import_torch

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.annotations import override

import numpy as np
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import torch.nn.functional as F

from ray.rllib.models import ModelCatalog

# Centralised critic model takes agen'ts observation + others' actions as input

class CentralisedCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = int(np.prod(obs_space.shape))

        if num_outputs is None:
            num_outputs = int(np.prod(action_space.shape)) if hasattr(action_space, 'shape') else action_space.n
        self.num_outputs = num_outputs

        self.policy_branch = nn.Sequential(
            nn.Linear(self.obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

        self.value_branch = nn.Sequential(
            nn.Linear(self.obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        opponent_action_dim = model_config.get("custom_model_config", {}).get("opponent_action_dim", 0)
        self.central_vf_branch = nn.Sequential(
            nn.Linear(self.obs_size + opponent_action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs_flat"]
        self._value_out = self.value_branch(x)
        return self.policy_branch(x), []

    def value_function(self):
        return self._value_out.squeeze(-1)

    def central_value_function(self, obs, opponent_actions):
        x = torch.cat([obs, opponent_actions], dim=-1)
        return self.central_vf_branch(x).squeeze(-1)

OPPONENT_ACTIONS = "opponent_actions"

class centralisedValueMixin:
    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


def centralised_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    if other_agent_batches:
        # find minimal length across current and opponent batches
        cur_len = len(sample_batch[SampleBatch.ACTIONS])
        other_lens = [len(batch[SampleBatch.ACTIONS])
                      for _, (_, _, batch) in other_agent_batches.items()]
        min_len = min([cur_len] + other_lens)

        # trim the current sample_batch
        for key in sample_batch:
            sample_batch[key] = sample_batch[key][:min_len]

        # build opponent_actions of shape [min_len, num_opponents]
        all_opponent_actions = [
            np.array(batch[SampleBatch.ACTIONS][:min_len], dtype=np.float32)
            for _, (_, _, batch) in other_agent_batches.items()
        ]
        opponent_actions = torch.tensor(
            np.stack(all_opponent_actions, axis=1),
            dtype=torch.float32
        )
        sample_batch[OPPONENT_ACTIONS] = opponent_actions

        obs = torch.tensor(sample_batch[SampleBatch.CUR_OBS], dtype=torch.float32)
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(obs, opponent_actions)
                  .detach().cpu().numpy()
        )
    else:
        sample_batch[SampleBatch.VF_PREDS] = torch.zeros_like(
            torch.tensor(sample_batch[SampleBatch.REWARDS], dtype=torch.float32)
        ).numpy()

    # Compute GAE
    last_r = 0.0 if sample_batch[SampleBatch.TERMINATEDS][-1] else sample_batch[SampleBatch.VF_PREDS][-1]
    return compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"]
    )


def central_value_stats(policy, train_batch):
    return {
        "central_vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy.central_value_out
        )
    }

# Custom policy class for centralised critic
# This class inherits from PPOTorchPolicy and centralisedValueMixin
# It overrides the postprocess_trajectory and loss methods to include centralised critic functionality
# It also includes a stats function to compute explained variance of the centralised value function

class CCPPOPolicy(PPOTorchPolicy, centralisedValueMixin):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        centralisedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        return centralised_critic_postprocessing(self, sample_batch, other_agent_batches, episode)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        obs = train_batch[SampleBatch.CUR_OBS]
        if OPPONENT_ACTIONS in train_batch:
            opp_acts = train_batch[OPPONENT_ACTIONS]
            self.central_value_out = self.model.central_value_function(obs, opp_acts)
        else:
            self.central_value_out = torch.zeros(obs.shape[0])

        return super().loss(model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch):
        stats = super().stats_fn(train_batch)
        stats.update(central_value_stats(self, train_batch))
        return stats

ModelCatalog.register_custom_model("My_CC_PPO", CentralisedCriticModel)