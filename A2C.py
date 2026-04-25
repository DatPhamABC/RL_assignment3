import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time
import random
import matplotlib.pyplot as plt
from gymnasium.vector import SyncVectorEnv


class A2CNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value


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

        self.ac_net = A2CNetwork(
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

        logits, values = self.ac_net(s_tensor)

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions.cpu().numpy(), log_probs, values

    def update_batch(self, log_probs, values, returns):
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).sum()
        critic_loss = advantages.pow(2).sum()

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_returns(self, rewards):
        returns = []
        discount_return = 0

        for r in reversed(rewards):
            discount_return = r + self.gamma * discount_return
            returns.insert(0, discount_return)

        return torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )

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


def A2C_run(
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

    timestep = 0
    next_eval_timestep = eval_interval

    while timestep < n_timesteps:
        s, _ = envs.reset(seed=seed)

        episode_log_probs = [[] for _ in range(num_envs)]
        episode_rewards = [[] for _ in range(num_envs)]
        episode_values = [[] for _ in range(num_envs)]
        episode_lengths = np.zeros(num_envs)
        finished = np.zeros(num_envs, dtype=bool)

        while not finished.all() and timestep < n_timesteps:
            actions, log_probs, values = agent.select_action(s)

            s_next, rewards, terminated, truncated, _ = envs.step(actions)

            episode_lengths += 1

            done = (
                terminated
                | truncated
                | (episode_lengths >= max_episode_length)
            )

            for i in range(num_envs):
                if not finished[i]:
                    episode_log_probs[i].append(log_probs[i])
                    episode_values[i].append(values[i])
                    episode_rewards[i].append(rewards[i])

                    if done[i]:
                        finished[i] = True

            s = s_next
            timestep += num_envs

            if timestep >= next_eval_timestep:
                mean_return = agent.evaluate(
                    eval_envs,
                    n_eval_episodes=n_eval_episodes,
                    max_episode_length=max_episode_length
                )

                eval_timesteps.append(timestep)
                eval_returns.append(mean_return)

                print(
                    f"Timestep {timestep}/{n_timesteps} | "
                    f"Mean eval return: {mean_return:.2f}"
                )

                next_eval_timestep += eval_interval

        all_log_probs = []
        all_values = []
        all_returns = []

        for i in range(num_envs):
            returns = agent.calculate_returns(episode_rewards[i])

            all_log_probs.extend(episode_log_probs[i])
            all_values.extend(episode_values[i])
            all_returns.append(returns)

        all_returns = torch.cat(all_returns)

        agent.update_batch(
            all_log_probs,
            all_values,
            all_returns
        )

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

    eval_returns, eval_timesteps = A2C_run(
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
    plt.ylabel("Evaluation Return")
    plt.savefig("A2C.png")


if __name__ == "__main__":
    t0 = time.time()
    test()
    print(time.time() - t0)