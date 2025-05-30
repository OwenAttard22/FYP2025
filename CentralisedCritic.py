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

# Centralised critic model takes all agents' observations and actions as input

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

        # opponent_action_dim = model_config.get("custom_model_config", {}).get("opponent_action_dim", 0)
        # self.central_vf_branch = nn.Sequential(
            # nn.Linear(self.obs_size + opponent_action_dim, 128),
            
        cfg = model_config.get("custom_model_config", {})
        self.num_agents = cfg["num_agents"]
        self.action_size = cfg["action_size"]
        
        total_dim = (self.obs_size + self.action_size) * self.num_agents
        self.central_vf_branch = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
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
    
    def central_value_function_all(self, all_obs_flat, all_acts_flat):
        """ 
        all_obs_flat: [B, num_agents * obs_size]
        all_acts_flat: [B, num_agents * action_size]
        """
        device = next(self.parameters()).device
        # x = torch.cat([all_obs_flat, all_acts_flat], dim=-1)
        x = torch.cat([all_obs_flat.to(device), all_acts_flat.to(device)], dim=-1)
        return self.central_vf_branch(x).squeeze(-1)

OPPONENT_ACTIONS = "opponent_actions"

class centralisedValueMixin:
    def __init__(self):
        # self.compute_central_vf = self.model.central_value_function
        self.compute_central_vf = self.model.central_value_function_all


def centralised_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    try: # work with values we already have
        if other_agent_batches is not None:
            # find minimal length across current and opponent batches
            cur_len = len(sample_batch[SampleBatch.ACTIONS])
            other_lens = [len(batch[SampleBatch.ACTIONS])
                        for _, (_, _, batch) in other_agent_batches.items()]
            min_len = min([cur_len] + other_lens)
            
            other = [
                (agent_id, batch)
                for agent_id, (_, _, batch) in other_agent_batches.items()
            ]

            # trim the current sample_batch
            for key in sample_batch:
                sample_batch[key] = sample_batch[key][:min_len]

            # collect local
            self_acts = np.array(sample_batch[SampleBatch.ACTIONS], dtype=np.float32)[:min_len]
            self_obs = np.array(sample_batch[SampleBatch.CUR_OBS], dtype=np.float32)[:min_len]
            
            # collect others
            acts_list = [self_acts]
            obs_list = [self_obs]
            for _, batch in other:
                acts_list.append(np.array(batch[SampleBatch.ACTIONS], dtype=np.float32)[:min_len])
                obs_list.append(np.array(batch[SampleBatch.CUR_OBS], dtype=np.float32)[:min_len])
                
            # combine
            T = min_len
            k = len(acts_list)
            all_acts = np.stack(acts_list, axis=1)
            all_obs = np.stack(obs_list, axis=1)
            
            # pad zero to agents if k < N
            N = policy.model.num_agents
            if k < N:
                pad_agents = N - k
                all_acts = np.concatenate([
                    all_acts,
                    np.zeros((T, pad_agents), dtype=np.float32)
                ], axis=1)
                all_obs = np.concatenate([
                    all_obs,
                    np.zeros((T, pad_agents, policy.model.obs_size), dtype=np.float32)
                ], axis=1)

            # flatten
            all_acts_flat = torch.from_numpy(all_acts.reshape(T, -1))
            all_obs_flat = torch.from_numpy(all_obs.reshape(T, -1))
            
            # compute central value function
            sample_batch[SampleBatch.VF_PREDS] = (
                policy.model.central_value_function_all(all_obs_flat, all_acts_flat).detach().cpu().numpy()
            )
            
        else:
            sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
                sample_batch[SampleBatch.REWARDS], dtype=np.float32
            )
            
    except ValueError as e:
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )
        
    terminateds = sample_batch[SampleBatch.TERMINATEDS]
    last_pred = sample_batch[SampleBatch.VF_PREDS][-1]
    last_r = 0.0 if terminateds[-1] else last_pred
    
    return compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )


def central_value_stats(policy, train_batch):
    targets = train_batch[Postprocessing.VALUE_TARGETS]
    preds = policy.central_value_out

    targets = targets.detach().cpu()
    preds = preds.detach().cpu()

    return {
        "central_vf_explained_var": explained_variance(targets, preds)
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
        self.central_value_out = torch.as_tensor(
            train_batch[SampleBatch.VF_PREDS],
            dtype=torch.float32,
            device=obs.device
        )

        return super().loss(model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch):
        stats = super().stats_fn(train_batch)
        stats.update(central_value_stats(self, train_batch))
        return stats

ModelCatalog.register_custom_model("My_CC_PPO", CentralisedCriticModel)