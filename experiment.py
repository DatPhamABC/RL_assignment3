import os
import time
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from policy_network import reinforce_run
from actor_critic import actor_critic_run
from A2C import A2C_run


# ============================================================
# PARAMETERS
# ============================================================

N_TIMESTEPS = 1_000_000
MAX_EPISODE_LENGTH = 500

GAMMA = 0.99
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128

NUM_ENVS = 20

EVAL_INTERVAL = 20000
N_EVAL_EPISODES = 30

SEEDS = [42, 123, 456, 789, 1000]

PLOT = False

SAVE_DIR = "cartpole_comparison_results"
PLOT_FILENAME = "cartpole_algorithm_comparison.png"

PRINT_EVERY_SEED = True


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():
    os.makedirs(SAVE_DIR, exist_ok=True)

    algorithms = {
        "REINFORCE": reinforce_run,
        "Actor-Critic": actor_critic_run,
        "A2C": A2C_run,
    }

    all_results = {}

    total_runs = len(algorithms) * len(SEEDS)
    completed_runs = 0
    experiment_start = time.time()

    for alg_name, run_fn in algorithms.items():
        print(f"\n==============================")
        print(f"Running {alg_name}")
        print(f"==============================")

        seed_returns = []
        seed_timesteps = []

        alg_start = time.time()

        for seed in SEEDS:
            seed_start = time.time()

            print(f"\nStarting {alg_name}, seed {seed}")
            print(f"Progress: {completed_runs}/{total_runs} runs completed")

            eval_returns, eval_timesteps = run_fn(
                n_timesteps=N_TIMESTEPS,
                max_episode_length=MAX_EPISODE_LENGTH,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                hidden_dim=HIDDEN_DIM,
                plot=PLOT,
                eval_interval=EVAL_INTERVAL,
                n_eval_episodes=N_EVAL_EPISODES,
                num_envs=NUM_ENVS,
                seed=seed,
            )

            seed_returns.append(eval_returns)
            seed_timesteps.append(eval_timesteps)

            completed_runs += 1
            seed_time = time.time() - seed_start
            total_time = time.time() - experiment_start

            print(f"Finished {alg_name}, seed {seed}")
            print(f"Seed runtime: {seed_time:.2f} seconds")
            print(f"Total elapsed time: {total_time:.2f} seconds")
            print(f"Progress: {completed_runs}/{total_runs} runs completed")

        alg_time = time.time() - alg_start
        print(f"\nFinished all seeds for {alg_name}")
        print(f"{alg_name} runtime: {alg_time:.2f} seconds")

        all_results[alg_name] = {
            "returns": np.array(seed_returns),
            "timesteps": seed_timesteps[0],
        }

    return all_results


def save_results(all_results):
    os.makedirs(SAVE_DIR, exist_ok=True)

    raw_path = os.path.join(SAVE_DIR, "raw_cartpole_comparison_results.npz")

    save_dict = {
        "seeds": np.array(SEEDS),
        "n_timesteps": np.array(N_TIMESTEPS),
        "max_episode_length": np.array(MAX_EPISODE_LENGTH),
        "gamma": np.array(GAMMA),
        "learning_rate": np.array(LEARNING_RATE),
        "hidden_dim": np.array(HIDDEN_DIM),
        "num_envs": np.array(NUM_ENVS),
        "eval_interval": np.array(EVAL_INTERVAL),
        "n_eval_episodes": np.array(N_EVAL_EPISODES),
    }

    summary = {}

    for alg_name, result in all_results.items():
        safe_name = alg_name.lower().replace("-", "_").replace(" ", "_")

        returns = result["returns"]
        timesteps = result["timesteps"]

        mean_returns = returns.mean(axis=0)
        std_returns = returns.std(axis=0)

        save_dict[f"{safe_name}_returns"] = returns
        save_dict[f"{safe_name}_timesteps"] = timesteps
        save_dict[f"{safe_name}_mean_returns"] = mean_returns
        save_dict[f"{safe_name}_std_returns"] = std_returns

        np.savez(
            os.path.join(SAVE_DIR, f"{safe_name}_results.npz"),
            returns=returns,
            timesteps=timesteps,
            mean_returns=mean_returns,
            std_returns=std_returns,
            seeds=np.array(SEEDS),
        )

        summary[alg_name] = {
            "final_mean_return": float(mean_returns[-1]),
            "final_std_return": float(std_returns[-1]),
            "best_mean_return": float(mean_returns.max()),
            "best_timestep": int(timesteps[np.argmax(mean_returns)]),
        }

    np.savez(raw_path, **save_dict)

    config_path = os.path.join(SAVE_DIR, "experiment_config_and_summary.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "config": {
                    "n_timesteps": N_TIMESTEPS,
                    "max_episode_length": MAX_EPISODE_LENGTH,
                    "gamma": GAMMA,
                    "learning_rate": LEARNING_RATE,
                    "hidden_dim": HIDDEN_DIM,
                    "num_envs": NUM_ENVS,
                    "eval_interval": EVAL_INTERVAL,
                    "n_eval_episodes": N_EVAL_EPISODES,
                    "seeds": SEEDS,
                },
                "summary": summary,
            },
            f,
            indent=4,
        )

    print(f"\nSaved raw data to: {raw_path}")
    print(f"Saved config and summary to: {config_path}")


def plot_results(all_results):
    plt.figure(figsize=(12, 8))

    for alg_name, result in all_results.items():
        returns = result["returns"]
        timesteps = result["timesteps"]

        mean_returns = returns.mean(axis=0)
        std_returns = returns.std(axis=0)

        plt.plot(
            timesteps,
            mean_returns,
            linewidth=3,
            label=alg_name
        )

        plt.fill_between(
            timesteps,
            mean_returns - std_returns,
            mean_returns + std_returns,
            alpha=0.2
        )

    plt.xlabel("Timesteps", fontsize=22)
    plt.ylabel("Evaluation return", fontsize=22)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)

    path = os.path.join(SAVE_DIR, PLOT_FILENAME)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {path}")


if __name__ == "__main__":
    t0 = time.time()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    results = run_experiment()
    save_results(results)
    plot_results(results)

    print(f"\nTotal runtime: {time.time() - t0:.2f} seconds")