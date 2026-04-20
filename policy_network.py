
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time
from gymnasium.vector import SyncVectorEnv
import matplotlib.pyplot as plt
from collections import deque
import random


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        learning_rate,
        gamma,
        hidden_dim=128,
        use_target_network=False,
        target_update_freq=1000,
        device=None
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.use_target_network = use_target_network
        self.target_update_freq = target_update_freq
        self.step_count = 0

        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.pol_net = PolicyNetwork(obs_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.pol_net.parameters(), lr=learning_rate)
        self.loss = None

    def _to_tensor(self, s):
        return torch.from_numpy(np.asarray(s, dtype=np.float32)).to(self.device)

    def select_action(self, s, temp=None):
        s_tensor = self._to_tensor(s)
        s_tensor = s_tensor.unsqueeze(0)

        if temp is None:
            raise KeyError("Provide a temperature")

        # print(q_values)
        logits = self.pol_net(s_tensor)
        # print(logits)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        log_prob = dist.log_prob(a)

        return a.item(), log_prob

    def update_batch(self, log_probs, returns):
       

        loss = []
        for log_prob, G_t in zip(log_probs, returns):
            loss.append(-log_prob*G_t)
        
        self.optimizer.zero_grad()
        loss = torch.stack(loss).sum()
        loss.backward()
        self.optimizer.step()

    def calculate_returns(self, rewards):
        returns = []
        discount_return = 0
        for r in reversed(rewards):
            discount_return = r + self.gamma*discount_return
            returns.insert(0, discount_return)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns
    

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=500):
        returns = []
        for _ in range(n_eval_episodes):
            s, _ = eval_env.reset()
            R_ep = 0

            for _ in range(max_episode_length):
                s_tensor = self._to_tensor(s).unsqueeze(0)

                probs = self.pol_net(s_tensor)
                a = torch.argmax(probs, dim=1).item()
                # a = self.select_action(s)
                s_prime, r, terminated, truncated, _ = eval_env.step(a)
                done = terminated or truncated
                R_ep += r

                if done:
                    break

                s = s_prime

            returns.append(R_ep)

        return np.mean(returns)


def run(
    n_timesteps,
    max_episode_length,
    learning_rate,
    gamma,
    policy='egreedy',
    epsilon=None,
    temp=None,
    hidden_dim=128,
    env_steps_per_update=100,
    plot=False,
    eval_interval=500,
    n_eval_episodes=10,
    seed=None
):
    """
    Runs a single repetition of a DQN agent.
    Returns:
        eval_returns: array of evaluation returns
        eval_timesteps: array of timesteps at which evaluation happened
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    env = gym.make("CartPole-v1", render_mode="human" if plot else None)
    eval_env = gym.make("CartPole-v1")

    # num_envs = 20
    # envs = SyncVectorEnv([make_env for _ in range(num_envs)])

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    pi = Agent(
        obs_dim,
        n_actions,
        learning_rate,
        gamma,
        hidden_dim=hidden_dim
    )

    eval_timesteps = []
    eval_returns = []

    for timestep in range(n_timesteps):
        s, _ = env.reset(seed=seed)

        episode_log_probs = []
        episode_rewards = []

        done = False
        while not done:
            a, log_prob = pi.select_action(s, temp=temp)

            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated | truncated
            # print(done)

            episode_log_probs.append(log_prob)
            episode_rewards.append(r)

            s = s_next

        if timestep % eval_interval == 0:
            mean_return = pi.evaluate(
                eval_env,
                n_eval_episodes=n_eval_episodes,
                max_episode_length=max_episode_length
            )
            eval_timesteps.append(timestep)
            eval_returns.append(mean_return)
        
        returns = pi.calculate_returns(episode_rewards)
        pi.update_batch(episode_log_probs, returns=returns)


    env.close()
    env.close()
    eval_env.close()

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1000
    max_episode_length = 800
    eval_interval = 100
    n_eval_episodes = 10

    gamma = 0.99
    learning_rate = 1e-3

    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0

    plot = False

    eval_returns, eval_timesteps = run(
        n_timesteps=n_timesteps,
        max_episode_length=max_episode_length,
        learning_rate=learning_rate,
        gamma=gamma,
        policy=policy,
        epsilon=epsilon,
        temp=temp,
        eval_interval = eval_interval,
        n_eval_episodes = n_eval_episodes,
        plot=plot
    )

    print("Evaluation timesteps:", eval_timesteps)
    print("Evaluation returns:", eval_returns)

    plt.plot(eval_timesteps, eval_returns)
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation return")
    plt.savefig('reinforce.png')


if __name__ == '__main__':
    t0 = time.time()
    test()
    print(time.time() - t0)