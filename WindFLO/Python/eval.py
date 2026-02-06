from wind_scenario import WindScenario
from KusiakEnergyEvaluator import KusiakEnergyEvaluator
from hill_climb import hill_climb
from genetic import genetic_algorithm
from plot_turbines import plot_turbines

def run_scenario_ga(scenario_path: str, seed: int | None = None) -> None:
    scenario = WindScenario(scenario_path)
    evaluator = KusiakEnergyEvaluator(scenario)

    print("=" * 50)
    print(f"SCENARIO: {scenario_path}")
    print("=" * 70)
    print(f"number of turbines   : {scenario.nturbines}")
    print(f"farm size            : {scenario.width} x {scenario.height}")
    print(f"obstacles            : {scenario.obstacles.shape[0]}")
    print(f"min spacing (sq)     : {scenario.minDist}")
    print("-" * 50)

    best_layout, best_cost = genetic_algorithm(
        scenario,
        population_size=10,
        generations=2,
        elite_count=2,
        step_size=50.0,
        crossover_imports=10,
        mutation_count=3,
        log_every=1,
        seed=seed,
    )

    #re-evaluate
    final_cost = float(evaluator.evaluate(best_layout))


    print("-" * 50)
    print("FINAL RESULT")
    print("-" * 70)
    print(f"best cost (reported) : {best_cost:.6g}")
    print(f"best cost (re-eval)  : {final_cost:.6g}")
    print(f"layout               : {best_layout.shape}")
    print("=" * 50)
    print()

    plot_turbines(scenario, best_layout, "optimised wind farm turbine layout")

def run_scenario_hill_climb(scenario_path: str, seed: int | None = None) -> None:
    """
   load scenario and run hill-climber on it. print results
    """


    scenario = WindScenario(scenario_path)
    evaluator = KusiakEnergyEvaluator(scenario)

    print("=" * 50)
    print(f"SCENARIO: {scenario_path}")
    print(f"seed: {seed}")
    print("=" * 70)
    print(f"number of turbines : {scenario.nturbines}")
    print(f"farm size          : {scenario.width} x {scenario.height}")
    print(f"min spacing (sq)   : {scenario.minDist}")
    print(f"obstacles          : {scenario.obstacles.shape[0]}")
    print("-" * 50)

    # Run prototype hill climber
    best_layout, best_cost = hill_climb(
        scenario,
        iterations=10,                 # small for prototype
        step_size=50.0,
        max_attempts_per_iteration=20,
        log_every=1,
        seed = seed
    )

    #re-evaluate
    final_cost = float(evaluator.evaluate(best_layout))

    print("-" * 50)
    print("FINAL RESULT")
    print("-" * 70)
    print(f"best cost (reported) : {best_cost:.6g}")
    print(f"best cost (re-eval)  : {final_cost:.6g}")
    print(f"layout shape         : {best_layout.shape}")
    print("=" * 50)
    print()

    plot_turbines(scenario, best_layout, "optimised layout")


def main():
    print("\nHILL CLIMBER - MULTI-SCENARIO RUN\n")


    seed = [69, 420, 2810]
    scenario_paths = ["../Scenarios/01.xml", "../Scenarios/obs_05.xml"]

    for seeds in seed:
        for scenario_path in scenario_paths:
            run_scenario_hill_climb(scenario_path, seeds)

    # # without obstacles
    # run_scenario_hill_climb("../Scenarios/01.xml")
    # #with obstacles
    # run_scenario_hill_climb("../Scenarios/obs_05.xml")

    # print("\nGENETIC ALGORITHM - MULTI-SCENARIO RUN\n")
    # # without obstacles
    # run_scenario_ga("../Scenarios/01.xml")
    # #with obstacles
    # run_scenario_ga("../Scenarios/obs_05.xml")





if __name__ == "__main__":
    main()