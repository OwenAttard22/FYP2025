import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim) # categorical istribution for discrete action space
        )

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).detach()
        return action, log_prob, dist

    
    
class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim_per_agent: int, act_dim_per_agent: int, num_agents: int):
        super(CentralizedCritic, self).__init__()
        input_dim = num_agents * (obs_dim_per_agent + act_dim_per_agent)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)  # Outputs a single scalar value
        )

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor):
        """
        Inputs:
            all_obs: (batch_size, num_agents, obs_dim)
            all_actions: (batch_size, num_agents, act_dim) ← one-hot if discrete
        """
        x = torch.cat([all_obs, all_actions], dim=-1)   # (batch_size, num_agents, obs+act)
        x = x.view(x.size(0), -1)  # Flatten agent dimension
        return self.net(x)  # Output: (batch_size, 1)


class RolloutBuffer:
    def __init__(self, num_steps, num_agents, obs_dim, act_dim):
        self.num_steps = num_steps
        self.ptr = 0

        self.obs = torch.zeros((num_steps, num_agents, obs_dim))
        self.actions = torch.zeros((num_steps, num_agents), dtype=torch.long)
        self.log_probs = torch.zeros((num_steps, num_agents))
        self.rewards = torch.zeros((num_steps, num_agents))
        self.dones = torch.zeros((num_steps, num_agents))
        self.values = torch.zeros((num_steps, 1))  # Centralized value
        self.advantages = torch.zeros((num_steps, 1))
        self.returns = torch.zeros((num_steps, 1))

    def add(self, obs, actions, log_probs, rewards, dones, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        gae = 0
        for step in reversed(range(self.ptr)):
            next_value = last_value if step == self.ptr - 1 else self.values[step + 1]
            delta = self.rewards[step].mean() + gamma * next_value * (1 - self.dones[step].mean()) - self.values[step]
            gae = delta + gamma * lam * (1 - self.dones[step].mean()) * gae
            self.advantages[step] = gae
            self.returns[step] = self.advantages[step] + self.values[step]

    def get(self):
        return (
            self.obs[:self.ptr],
            self.actions[:self.ptr],
            self.log_probs[:self.ptr],
            self.advantages[:self.ptr],
            self.returns[:self.ptr],
        )

    def reset(self):
        self.ptr = 0
        
        
class MAPPO:
    def __init__(
        self,
        actor,
        critic,
        actor_lr=3e-4,
        critic_lr=3e-4,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    ):
        self.actor = actor
        self.critic = critic
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    def ppo_update(self, buffer, batch_size=64, epochs=4):
        obs, actions, old_log_probs, advantages, returns = buffer.get()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset_size = obs.shape[0]

        for _ in range(epochs):
            for i in range(0, dataset_size, batch_size):
                end = i + batch_size
                batch_obs = obs[i:end]  # (B, N, obs_dim)
                batch_actions = actions[i:end]
                batch_log_probs_old = old_log_probs[i:end]
                batch_adv = advantages[i:end]
                batch_returns = returns[i:end]

                # ========== ACTOR ==========
                dist = self.actor(batch_obs.view(-1, batch_obs.size(-1)))  # (B*N, act_dim)
                new_log_probs = dist.log_prob(batch_actions.view(-1)).view_as(batch_log_probs_old)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - batch_log_probs_old).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                # ========== CRITIC ==========
                # Recompute obs + actions for critic separately and with no grad attached
                with torch.no_grad():
                    dist_critic = self.actor(batch_obs.view(-1, batch_obs.size(-1)))
                    action_one_hot = F.one_hot(batch_actions.view(-1), num_classes=dist_critic.probs.shape[-1]).float()
                    action_one_hot = action_one_hot.view(batch_actions.shape[0], batch_actions.shape[1], -1)  # (B, N, act_dim)

                values = self.critic(batch_obs, action_one_hot).squeeze(-1)
                critic_loss = F.mse_loss(values, batch_returns.squeeze(-1))

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

