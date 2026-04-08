from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



# set a clean default style
sns.set_theme(style="whitegrid")


def ensure_figure_dir(base_dir: str) -> Path:
    """
    create the figures output directory if needed
    """
    figure_dir = Path(base_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def load_results(summary_csv: str, history_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    load summary and convergence-history csv files
    """
    summary_df = pd.read_csv(summary_csv)
    history_df = pd.read_csv(history_csv)
    return summary_df, history_df


def get_single_scenario_name(summary_df: pd.DataFrame) -> str:
    """
    get the scenario name for figure titles

    csv contains one scenario only
    """
    unique_scenarios = summary_df["scenario"].unique()
    if len(unique_scenarios) == 1:
        return str(unique_scenarios[0])
    return "multiple scenarios"


def add_scaled_cost_columns(
        summary_df: pd.DataFrame,
        history_df: pd.DataFrame,
        scale_factor: float = 1e6,
) -> None:
    """
    add scaled cost columns so very small energy-cost values are easier to read on plots

    default scaling:
    0.000659 -> 659.0
    """
    summary_df["best_cost_scaled"] = summary_df["best_cost"] * scale_factor
    history_df["best_cost_so_far_scaled"] = history_df["best_cost_so_far"] * scale_factor


def interpolate_run_history(group: pd.DataFrame, common_x: np.ndarray) -> np.ndarray:
    """
    interpolate one run's convergence history onto a common evaluation axis
    """
    group = group.sort_values("evaluation")
    x = group["evaluation"].to_numpy()
    y = group["best_cost_so_far_scaled"].to_numpy()
    return np.interp(common_x, x, y)


def plot_convergence_per_run_by_algorithm(
        history_df: pd.DataFrame,
        scenario_name: str,
        figure_dir: Path,
        filename: str = "convergence_per_run_by_algorithm.png",
) -> None:
    """
    plot best-so-far cost against evaluation count for each run,
    colouring and grouping by algorithm
    """
    algorithms = history_df["algorithm"].unique()

    plt.figure(figsize=(10, 6))

    for algorithm in algorithms:
        algo_df = history_df[history_df["algorithm"] == algorithm]

        for _, group in algo_df.groupby("run_index"):
            group = group.sort_values("evaluation")
            seed = group["seed"].iloc[0]
            label = f"{algorithm.upper()} seed {seed}"
            plt.plot(group["evaluation"], group["best_cost_so_far_scaled"], label=label)

    plt.xlabel("Objective-function evaluations")
    plt.ylabel("Best-so-far energy cost (x 10^-6)")
    plt.title(f"Convergence by Evaluation Calls for Individual Runs ({scenario_name})")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mean_convergence_by_algorithm(
        history_df: pd.DataFrame,
        scenario_name: str,
        figure_dir: Path,
        filename: str = "mean_convergence_by_algorithm.png",
) -> None:
    """
    plot one mean best-so-far convergence curve per algorithm
    """
    algorithms = history_df["algorithm"].unique()

    plt.figure(figsize=(10, 6))

    for algorithm in algorithms:
        algo_df = history_df[history_df["algorithm"] == algorithm]
        grouped = list(algo_df.groupby("run_index"))

        if not grouped:
            continue

        max_common_eval = min(group["evaluation"].max() for _, group in grouped)
        common_x = np.arange(1, max_common_eval + 1)

        interpolated_runs = []
        for _, group in grouped:
            interpolated_runs.append(interpolate_run_history(group, common_x))

        interpolated_runs = np.array(interpolated_runs)
        mean_curve = np.mean(interpolated_runs, axis=0)
        std_curve = np.std(interpolated_runs, axis=0)

        plt.plot(common_x, mean_curve, label=algorithm.upper())
        plt.fill_between(
            common_x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
        )

    plt.xlabel("Objective-function evaluations")
    plt.ylabel("Best-so-far energy cost (x 10^-6)")
    plt.title(f"Mean Convergence by Evaluation Calls ({scenario_name})")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_final_cost_violin(
        summary_df: pd.DataFrame,
        scenario_name: str,
        figure_dir: Path,
        filename: str = "final_cost_violin.png",
) -> None:
    """
    violin plot of final best cost by algorithm
    """
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=summary_df, x="algorithm", y="best_cost_scaled", inner="box")
    plt.xlabel("Algorithm")
    plt.ylabel("Final best energy cost (x 10^-6)")
    plt.title(f"Distribution of Final Energy Cost by Algorithm ({scenario_name})")
    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_runtime_violin(
        summary_df: pd.DataFrame,
        scenario_name: str,
        figure_dir: Path,
        filename: str = "runtime_violin.png",
) -> None:
    """
    violin plot of runtime by algorithm
    """
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=summary_df, x="algorithm", y="runtime_seconds", inner="box")
    plt.xlabel("Algorithm")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime Distribution by Algorithm ({scenario_name})")
    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def print_summary_statistics(summary_df: pd.DataFrame) -> None:
    """
    print simple summary statistics to the terminal
    """
    stats = summary_df.groupby("algorithm")[["best_cost", "runtime_seconds", "total_evaluations"]].agg(
        ["mean", "std", "min", "max"]
    )

    print("\nSummary statistics:")
    print(stats.to_string())


def generate_all_plots(results_dir: str = "Results") -> None:
    """
    load saved results and generate all figures
    """
    results_dir = Path(results_dir)
    summary_csv = results_dir / "summary" / "experiment_summary.csv"
    history_csv = results_dir / "history" / "experiment_history.csv"

    figure_dir = ensure_figure_dir(str(results_dir))

    summary_df, history_df = load_results(str(summary_csv), str(history_csv))
    scenario_name = get_single_scenario_name(summary_df)

    add_scaled_cost_columns(summary_df, history_df, scale_factor=1e6)

    print(f"loaded summary rows: {len(summary_df)}")
    print(f"loaded history rows: {len(history_df)}")
    print(f"scenario: {scenario_name}")

    plot_convergence_per_run_by_algorithm(history_df, scenario_name, figure_dir)
    plot_mean_convergence_by_algorithm(history_df, scenario_name, figure_dir)
    plot_final_cost_violin(summary_df, scenario_name, figure_dir)
    plot_runtime_violin(summary_df, scenario_name, figure_dir)

    print_summary_statistics(summary_df)

    print("\nFigures saved to:")
    print(figure_dir / "convergence_per_run_by_algorithm.png")
    print(figure_dir / "mean_convergence_by_algorithm.png")
    print(figure_dir / "final_cost_violin.png")
    print(figure_dir / "runtime_violin.png")


def main() -> None:
    generate_all_plots(results_dir="Results")


