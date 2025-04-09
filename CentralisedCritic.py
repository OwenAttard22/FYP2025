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
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.policy_net = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name + "_policy"
        )

        # centralised critic: obs + opponent_actions
        input_size = obs_space.shape[0] + model_config.get("opponent_action_len", 1)
        hidden_size = 256
        self.central_critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        return self.policy_net.forward(input_dict, state, seq_lens)

    def central_value_function(self, obs, opponent_actions):
        x = torch.cat([obs, opponent_actions], dim=-1)
        return self.central_critic(x).squeeze(-1)

OPPONENT_ACTIONS = "opponent_actions"

class centralisedValueMixin:
    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


def centralised_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    if other_agent_batches:
        all_opponent_actions = []
        for other_id, (_, _, batch) in other_agent_batches.items():
            all_opponent_actions.append(batch[SampleBatch.ACTIONS])

        # Concatenate all opponents' actions (assuming Discrete)
        opponent_actions = torch.stack([
            torch.tensor(a, dtype=torch.float32) for a in zip(*all_opponent_actions)
        ], dim=1)

        sample_batch[OPPONENT_ACTIONS] = opponent_actions

        # centralised value function estimate
        obs = torch.tensor(sample_batch[SampleBatch.CUR_OBS], dtype=torch.float32)
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(obs, opponent_actions).detach().cpu().numpy()
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

class CCPPOPolicy(centralisedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        centralisedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        return centralised_critic_postprocessing(self, sample_batch, other_agent_batches, episode)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        obs = train_batch[SampleBatch.CUR_OBS]
        opp_acts = train_batch[OPPONENT_ACTIONS]
        self.central_value_out = self.model.central_value_function(obs, opp_acts)
        return super().loss(model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch):
        stats = super().stats_fn(train_batch)
        stats.update(central_value_stats(self, train_batch))
        return stats

class CentralisedLSTMModel(RecurrentNetwork, TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralisedLSTMModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.cell_size = 64  # LSTM hidden size
        self.obs_size = int(np.product(obs_space.shape))
        self.num_outputs = num_outputs

        # Policy LSTM branch
        self.fc = nn.Linear(self.obs_size, 128)
        self.lstm = nn.LSTM(self.fc.out_features, self.cell_size, batch_first=True)
        self.policy_out = nn.Linear(self.cell_size, num_outputs)
        self.value_branch = nn.Linear(self.cell_size, 1)

        # centralised value function branch (different input: own obs + opponent actions)
        central_input_size = self.obs_size + model_config.get("opponent_action_dim", 0)
        self.central_vf_branch = nn.Sequential(
            nn.Linear(central_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._value_out = None

    @override(ModelV2)
    def get_initial_state(self):
        # Return a list of zeros for h and c
        h = [torch.zeros(1, self.cell_size), torch.zeros(1, self.cell_size)]
        return h

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):
        x = F.relu(self.fc(input_dict["obs_flat"]))
        x = x.unsqueeze(1)  # Add time dimension (B, 1, features)
        h0, c0 = state
        lstm_out, (h1, c1) = self.lstm(x, (h0.unsqueeze(0), c0.unsqueeze(0)))
        out = self.policy_out(lstm_out.squeeze(1))
        self._value_out = self.value_branch(lstm_out.squeeze(1))
        return out, [h1.squeeze(0), c1.squeeze(0)]

    @override(ModelV2)
    def value_function(self):
        return self._value_out.squeeze(1)

    def central_value_function(self, obs, opponent_actions):
        # obs: (B, obs_dim)
        # opponent_actions: (B, opponent_action_dim)
        x = torch.cat([obs, opponent_actions], dim=-1)
        return self.central_vf_branch(x).squeeze(1)
    
ModelCatalog.register_custom_model("My_CC_LSTM_PPO", CentralisedLSTMModel)