import os
import numpy as np
import matplotlib.pyplot as plt


# PARAMETERS

SAVE_DIR = "cartpole_comparison_results"

RAW_DATA_FILE = "actor_critic_mc_vs_td_results.npz"

PLOT_FILENAME = "replot_full_experiment_actor_critic_MC.png"

ALGORITHMS = [
    "REINFORCE",
    "A2C with advantage",
    "Actor-Critic MC",
]


def safe_name(alg_name):
    return (
        alg_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_results():
    path = os.path.join(SAVE_DIR, RAW_DATA_FILE)
    data = np.load(path)

    all_results = {}

    for alg_name in ALGORITHMS:
        name = safe_name(alg_name)

        returns_key = f"{name}_returns"
        timesteps_key = f"{name}_timesteps"

        if returns_key not in data or timesteps_key not in data:
            print(f"Skipping {alg_name}: not found in {path}")
            continue

        all_results[alg_name] = {
            "returns": data[returns_key],
            "timesteps": data[timesteps_key],
        }

    return all_results


def plot_results(all_results):
    plt.figure(figsize=(12, 8))

    DISPLAY_NAMES = {
        "REINFORCE": "REINFORCE",
        "A2C with advantage": "A2C",
        "Actor-Critic MC": "Actor-Critic",
    }

    for alg_name, result in all_results.items():
        returns = result["returns"]
        timesteps = result["timesteps"]

        mean_returns = returns.mean(axis=0)
        std_returns = returns.std(axis=0)

        plt.plot(
            timesteps,
            mean_returns,
            linewidth=3,
            label=DISPLAY_NAMES.get(alg_name, alg_name)
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
    results = load_results()
    plot_results(results)