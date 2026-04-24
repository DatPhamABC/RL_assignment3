import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time
import random
import matplotlib.pyplot as plt
from gymnasium.vector import SyncVectorEnv


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        features = self.shared(x)

        logits = self.actor(features)
        q_values = self.critic(features)

        return logits, q_values


class Agent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        learning_rate,
        gamma,
        hidden_dim=128,
        device=None
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.ac_net = ActorCriticNetwork(
            obs_dim,
            n_actions,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.ac_net.parameters(),
            lr=learning_rate
        )

    def _to_tensor(self, s):
        return torch.from_numpy(
            np.asarray(s, dtype=np.float32)
        ).to(self.device)

    def select_action(self, s):
        s_tensor = self._to_tensor(s)

        if s_tensor.ndim == 1:
            s_tensor = s_tensor.unsqueeze(0)

        logits, q_values = self.ac_net(s_tensor)

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        q_values_selected = q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        return actions.cpu().numpy(), log_probs, q_values_selected

    def update_td_batch(self, log_probs, q_values, rewards, next_states, dones):
        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        next_state_tensor = self._to_tensor(next_states)

        with torch.no_grad():
            next_logits, next_q_values = self.ac_net(next_state_tensor)
            next_dist = torch.distributions.Categorical(logits=next_logits)
            next_probs = next_dist.probs

            expected_next_q = (next_probs * next_q_values).sum(dim=1)

            td_targets = rewards + self.gamma * expected_next_q * (1 - dones)

        td_errors = td_targets - q_values

        actor_loss = -(log_probs * q_values.detach()).mean()
        critic_loss = td_errors.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, eval_envs, n_eval_episodes=30, max_episode_length=500):
        returns = []

        num_envs = eval_envs.num_envs
        s, _ = eval_envs.reset()

        episode_returns = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs)

        while len(returns) < n_eval_episodes:
            s_tensor = self._to_tensor(s)

            with torch.no_grad():
                logits, _ = self.ac_net(s_tensor)

            actions = torch.argmax(logits, dim=1).cpu().numpy()

            s_next, rewards, terminated, truncated, _ = eval_envs.step(actions)

            episode_returns += rewards
            episode_lengths += 1

            done = (
                terminated
                | truncated
                | (episode_lengths >= max_episode_length)
            )

            for i in range(num_envs):
                if done[i]:
                    returns.append(episode_returns[i])
                    episode_returns[i] = 0
                    episode_lengths[i] = 0

                    if len(returns) >= n_eval_episodes:
                        break

            s = s_next

        return np.mean(returns[:n_eval_episodes])


def actor_critic_run(
    n_timesteps,
    max_episode_length,
    learning_rate,
    gamma,
    hidden_dim=128,
    plot=False,
    eval_interval=500,
    n_eval_episodes=10,
    num_envs=20,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def make_env():
        return gym.make("CartPole-v1")

    envs = SyncVectorEnv([make_env for _ in range(num_envs)])
    eval_envs = SyncVectorEnv([make_env for _ in range(num_envs)])

    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    agent = Agent(
        obs_dim,
        n_actions,
        learning_rate,
        gamma,
        hidden_dim=hidden_dim
    )

    eval_timesteps = []
    eval_returns = []

    s, _ = envs.reset(seed=seed)
    episode_lengths = np.zeros(num_envs)

    timestep = 0

    while timestep < n_timesteps:
        actions, log_probs, q_values = agent.select_action(s)

        s_next, rewards, terminated, truncated, _ = envs.step(actions)

        episode_lengths += 1

        done = (
            terminated
            | truncated
            | (episode_lengths >= max_episode_length)
        )

        agent.update_td_batch(
            log_probs=log_probs,
            q_values=q_values,
            rewards=rewards,
            next_states=s_next,
            dones=done
        )

        episode_lengths[done] = 0

        s = s_next
        timestep += num_envs

        if timestep % eval_interval < num_envs:
            mean_return = agent.evaluate(
                eval_envs,
                n_eval_episodes=n_eval_episodes,
                max_episode_length=max_episode_length
            )

            eval_timesteps.append(timestep)
            eval_returns.append(mean_return)

    envs.close()
    eval_envs.close()

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1_000_000
    max_episode_length = 500
    eval_interval = 500
    n_eval_episodes = 10
    num_envs = 20

    gamma = 0.99
    learning_rate = 1e-3

    eval_returns, eval_timesteps = actor_critic_run(
        n_timesteps=n_timesteps,
        max_episode_length=max_episode_length,
        learning_rate=learning_rate,
        gamma=gamma,
        eval_interval=eval_interval,
        n_eval_episodes=n_eval_episodes,
        num_envs=num_envs,
        plot=False
    )

    print("Evaluation timesteps:", eval_timesteps)
    print("Evaluation returns:", eval_returns)

    plt.plot(eval_timesteps, eval_returns)
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation return")
    plt.savefig("actor_critic.png")


if __name__ == "__main__":
    t0 = time.time()
    test()
    print(time.time() - t0)