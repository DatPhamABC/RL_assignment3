import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Change this import if your DQN file has a different name
from policy_network import reinforce_run


# ============================================================
# Settings
# ============================================================

base_output_dir = "experiment_results_policy_network"
experiment_name = "policy_network_results_mid_settings_full_rep"
output_dir = os.path.join(base_output_dir, experiment_name)

n_timesteps = 1_000_000
max_episode_length = 500
gamma = 0.99

policy = None
epsilon = None
temp = None

learning_rate = 5e-4
hidden_dim = 128

eval_interval = 10_000
n_eval_episodes = 10
num_runs = 5

base_seed = 42

# ============================================================
# Utility functions
# ============================================================

def summarize_results(eval_returns):
    """
    eval_returns shape: (num_runs, n_evals)
    """
    mean_curve = np.mean(eval_returns, axis=0)
    std_curve = np.std(eval_returns, axis=0)

    summary = {
        "mean_final_return": float(mean_curve[-1]),
        "std_final_return": float(std_curve[-1]),
        "mean_return_over_last_3_evals": float(np.mean(mean_curve[-3:])),
        "best_mean_return": float(np.max(mean_curve)),
    }
    return mean_curve, std_curve, summary


def plot_with_std(ax, x, mean, std):
    ax.plot(x, mean)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)


def run_single():
    all_eval_returns = []
    eval_timesteps_reference = None

    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f"  Run {run_idx + 1}/{num_runs} with seed={seed}")

        eval_returns, eval_timesteps = reinforce_run(n_timesteps=n_timesteps,
                                                     max_episode_length=max_episode_length,
                                                     learning_rate=learning_rate,
                                                     gamma=gamma,
                                                     hidden_dim=hidden_dim,
                                                     eval_interval = eval_interval,
                                                     n_eval_episodes = n_eval_episodes,
                                                     seed = seed)

        all_eval_returns.append(eval_returns)

        if eval_timesteps_reference is None:
            eval_timesteps_reference = eval_timesteps
        else:
            if not np.array_equal(eval_timesteps_reference, eval_timesteps):
                raise RuntimeError("Evaluation timesteps differ across runs.")

    all_eval_returns = np.array(all_eval_returns)
    mean_curve, std_curve, summary = summarize_results(all_eval_returns)

    result = {
        "eval_timesteps": eval_timesteps_reference,
        "eval_returns": all_eval_returns,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "summary": summary,
    }
    return result


def save_result(result, output_dir):
    filename = 'policy_network.npz'
    path = os.path.join(output_dir, filename)

    np.savez(
        path,
        eval_returns=result["eval_returns"],
        eval_timesteps=result["eval_timesteps"],
    )


def save_summary_json(results, output_dir):
    summary_dict = {
        "settings": {
            "n_timesteps": n_timesteps,
            "max_episode_length": max_episode_length,
            "gamma": gamma,
            "policy": policy,
            "epsilon": epsilon,
            "temp": temp,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "eval_interval": eval_interval,
            "n_eval_episodes": n_eval_episodes,
            "num_runs": num_runs,
            "base_seed": base_seed,
        },
        "results": []
    }

    for result in results:
        summary_dict["results"].append({
            "summary": result["summary"],
            "filename": 'policy_network.npz',
        })

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)


# ============================================================
# Plotting
# ============================================================

def plot_results(results, output_dir):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for result in sorted(results):
        plot_with_std(
            ax,
            result["eval_timesteps"],
            result["mean_curve"],
            result["std_curve"]
        )

    plt.xlabel("Environment timesteps")
    plt.ylabel("Mean evaluation return")
    plt.title("Policy Network")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "policy_network.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(output_dir, exist_ok=True)

    results = []

    result = run_single()
    results.append(result)
    save_result(result, output_dir)

    save_summary_json(results, output_dir)
    plot_results(results, output_dir)


if __name__ == "__main__":
    main()