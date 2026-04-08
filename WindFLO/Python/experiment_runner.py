from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from wind_scenario import WindScenario
from genetic import genetic_algorithm
from hill_climb import hill_climb, stochastic_hill_climb, hill_climb_random_restarts
from plot_results import generate_all_plots


@dataclass
class ExperimentConfig:
    """
    configuration for one batch of experiments
    """
    scenario_paths: list[str]
    seeds: list[int]
    output_dir: str = "Results_obs_05"
    save_history: bool = True
    run_plots_after: bool = True

    # algorithm switches
    run_ga: bool = True
    run_hc: bool = True
    run_shc: bool = True
    run_rrhc: bool = True

    # GA settings
    ga_population_size: int = 10
    ga_generations: int = 20
    ga_elite_count: int = 2
    ga_tournament_size: int = 3
    ga_step_size: float = 50.0
    ga_crossover_imports: int = 10
    ga_mutation_count: int = 10
    ga_log_every: int = 1

    # HC settings
    hc_iterations: int = 250
    hc_step_size: float = 50.0
    hc_max_attempts_per_iteration: int = 25
    hc_log_every: int = 10

    # SHC settings
    shc_iterations: int = 250
    shc_step_size: float = 50.0
    shc_log_every: int = 10

    # RRHC settings
    rrhc_restarts: int = 15
    rrhc_iterations: int = 250
    rrhc_step_size: float = 50.0
    rrhc_max_attempts_per_iteration: int = 25
    rrhc_log_every: int = 1


def ensure_output_dirs(base_dir: str) -> tuple[Path, Path]:
    """
    create output directories for summary tables and convergence histories
    """
    base = Path(base_dir)
    summary_dir = base / "summary"
    history_dir = base / "history"

    summary_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    return summary_dir, history_dir


def scenario_name_from_path(scenario_path: str) -> str:
    """
    extract the file name from a full scenario path
    """
    return Path(scenario_path).name


def write_csv(filepath: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary_row(
        *,
        run_index: int,
        algorithm: str,
        scenario_path: str,
        seed: int,
        result: Any,
        extra_params: dict[str, Any],
) -> dict[str, Any]:
    """
    create one summary row for one algorithm run
    """
    row: dict[str, Any] = {
        "run_index": run_index,
        "algorithm": algorithm,
        "scenario": scenario_name_from_path(scenario_path),
        "scenario_path": scenario_path,
        "seed": seed,
        "best_cost": result.best_cost,
        "total_evaluations": result.total_evals,
        "runtime_seconds": result.runtime,
    }

    row.update(extra_params)
    return row


def build_history_rows(
        *,
        run_index: int,
        algorithm: str,
        scenario_path: str,
        seed: int,
        result: Any,
) -> list[dict[str, Any]]:
    """
    create convergence-history rows for one run
    """
    rows: list[dict[str, Any]] = []

    for evaluation, best_cost_so_far in zip(result.eval_history, result.best_cost_history):
        rows.append({
            "run_index": run_index,
            "algorithm": algorithm,
            "scenario": scenario_name_from_path(scenario_path),
            "scenario_path": scenario_path,
            "seed": seed,
            "evaluation": evaluation,
            "best_cost_so_far": best_cost_so_far,
        })

    return rows


def get_algorithm_jobs(
        config: ExperimentConfig,
) -> list[tuple[str, Callable[..., Any], dict[str, Any]]]:
    """
    build the list of enabled algorithm jobs
    """
    jobs: list[tuple[str, Callable[..., Any], dict[str, Any]]] = []

    if config.run_ga:
        jobs.append((
            "ga",
            genetic_algorithm,
            {
                "population_size": config.ga_population_size,
                "generations": config.ga_generations,
                "elite_count": config.ga_elite_count,
                "tournament_size": config.ga_tournament_size,
                "step_size": config.ga_step_size,
                "crossover_imports": config.ga_crossover_imports,
                "mutation_count": config.ga_mutation_count,
                "log_every": config.ga_log_every,
                "record_mean": True,
            },
        ))

    if config.run_hc:
        jobs.append((
            "hc",
            hill_climb,
            {
                "iterations": config.hc_iterations,
                "step_size": config.hc_step_size,
                "max_attempts_per_iteration": config.hc_max_attempts_per_iteration,
                "log_every": config.hc_log_every,
            },
        ))

    if config.run_shc:
        jobs.append((
            "shc",
            stochastic_hill_climb,
            {
                "iterations": config.shc_iterations,
                "step_size": config.shc_step_size,
                "log_every": config.shc_log_every,
            },
        ))

    if config.run_rrhc:
        jobs.append((
            "rrhc",
            hill_climb_random_restarts,
            {
                "restarts": config.rrhc_restarts,
                "iterations": config.rrhc_iterations,
                "step_size": config.rrhc_step_size,
                "max_attempts_per_iteration": config.rrhc_max_attempts_per_iteration,
                "log_every": config.rrhc_log_every,
            },
        ))

    return jobs


def run_experiments(
        config: ExperimentConfig,
        summary_file: Path,
        history_file: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    run all enabled algorithms across all configured scenarios and seeds

    results saved after every completed run because evaluations are expensive
    """
    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    jobs = get_algorithm_jobs(config)
    total_runs = len(config.scenario_paths) * len(config.seeds) * len(jobs)
    run_counter = 0

    print("-" * 50)
    print(f"starting experiment batch: {total_runs} total runs")
    print("-" * 50)

    for scenario_path in config.scenario_paths:
        scenario = WindScenario(scenario_path)

        print("\n" + "-" * 50)
        print(f"SCENARIO: {scenario_path}")
        print("-" * 50)

        for algorithm_name, algorithm_function, algorithm_params in jobs:
            print(f"\n--- algorithm: {algorithm_name} ---")

            for seed in config.seeds:
                run_counter += 1
                print(f"[RUN {run_counter}/{total_runs}] {algorithm_name} seed={seed}")

                result = algorithm_function(
                    scenario,
                    seed=seed,
                    **algorithm_params,
                )

                summary_row = build_summary_row(
                    run_index=run_counter,
                    algorithm=algorithm_name,
                    scenario_path=scenario_path,
                    seed=seed,
                    result=result,
                    extra_params=algorithm_params,
                )
                summary_rows.append(summary_row)

                if config.save_history:
                    history_rows.extend(
                        build_history_rows(
                            run_index=run_counter,
                            algorithm=algorithm_name,
                            scenario_path=scenario_path,
                            seed=seed,
                            result=result,
                        )
                    )

                write_csv(summary_file, summary_rows)

                if config.save_history:
                    write_csv(history_file, history_rows)

                print(
                    f"completed | "
                    f"best_cost={result.best_cost:.12f} | "
                    f"evals={result.total_evals} | "
                    f"runtime={result.runtime:.2f}s"
                )

    return summary_rows, history_rows


def main() -> None:
    """
    run a batch of experiments and save the outputs
    """

    config = ExperimentConfig(
        scenario_paths=[
            "../Scenarios/obs_05.xml"
        ],
        seeds=[69, 420, 2810, 9999, 2026, 666, 777, 12345, 1612, 2712 ],
        output_dir="Results_obs_05",
        save_history=True,
        run_plots_after=True,

        #algorithm flags
        run_ga=True,
        run_hc=True,
        run_shc=True,
        run_rrhc=True,


        ga_population_size=8,
        ga_generations=8,
        ga_elite_count=2,
        ga_tournament_size=3,
        ga_step_size=50.0,
        ga_crossover_imports=10,
        ga_mutation_count=10,
        ga_log_every=0,

        hc_iterations=25,
        hc_step_size=50.0,
        hc_max_attempts_per_iteration=5,
        hc_log_every=0,

        shc_iterations=40,
        shc_step_size=50.0,
        shc_log_every=0,

        rrhc_restarts=4,
        rrhc_iterations=5,
        rrhc_step_size=50.0,
        rrhc_max_attempts_per_iteration=5,
        rrhc_log_every=0,
    )

    summary_dir, history_dir = ensure_output_dirs(config.output_dir)

    summary_file = summary_dir / "experiment_summary.csv"
    history_file = history_dir / "experiment_history.csv"

    summary_rows, history_rows = run_experiments(
        config=config,
        summary_file=summary_file,
        history_file=history_file,
    )

    print(f"summary rows written: {len(summary_rows)}")
    print(f"history rows written: {len(history_rows)}")
    print(f"Summary saved to: {summary_file}")

    if config.save_history:
        print(f"History saved to: {history_file}")

    if config.run_plots_after:
        print("\nGenerating plots...")
        generate_all_plots(results_dir=config.output_dir)
        print("Plots complete.")


if __name__ == "__main__":
    main()