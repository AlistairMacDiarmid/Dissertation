from wind_scenario import WindScenario
from KusiakEnergyEvaluator import KusiakEnergyEvaluator
from hill_climb import hill_climb
from genetic import genetic_algorithm

def run_scenario_ga(scenario_path: str) -> None:
    scenario = WindScenario(scenario_path)
    evaluator = KusiakEnergyEvaluator(scenario)

    print("=" * 70)
    print(f"SCENARIO: {scenario_path}")
    print("=" * 70)
    print(f"number of turbines   : {scenario.nturbines}")
    print(f"farm size            : {scenario.width} x {scenario.height}")
    print(f"obstacles            : {scenario.obstacles.shape[0]}")
    print(f"min spacing (sq)     : {scenario.minDist}")
    print("-" * 70)

    best_layout, best_cost = genetic_algorithm(
        scenario,
        population_size=10,
        generations=10,
        elite_count=2,
        step_size=50.0,
        crossover_imports=10,
        mutation_count=3,
        log_every=1,
    )

    #re-evaluate
    final_cost = float(evaluator.evaluate(best_layout))


    print("-" * 70)
    print("FINAL RESULT")
    print("-" * 70)
    print(f"best cost (reported) : {best_cost:.6g}")
    print(f"best cost (re-eval)  : {final_cost:.6g}")
    print(f"layout               : {best_layout.shape}")
    print("=" * 70)
    print()

def run_scenario_hill_climb(scenario_path: str) -> None:
    """
   load scenario and run hill-climber on it. print results
    """

    scenario = WindScenario(scenario_path)
    evaluator = KusiakEnergyEvaluator(scenario)

    print("=" * 70)
    print(f"SCENARIO: {scenario_path}")
    print("=" * 70)
    print(f"number of turbines : {scenario.nturbines}")
    print(f"farm size          : {scenario.width} x {scenario.height}")
    print(f"min spacing (sq)   : {scenario.minDist}")
    print(f"obstacles          : {scenario.obstacles.shape[0]}")
    print("-" * 70)

    # Run prototype hill climber
    best_layout, best_cost = hill_climb(
        scenario,
        iterations=50,                 # small for prototype
        step_size=50.0,
        max_attempts_per_iteration=20,
        log_every=10,
    )

    #re-evaluate
    final_cost = float(evaluator.evaluate(best_layout))

    print("-" * 70)
    print("FINAL RESULT")
    print("-" * 70)
    print(f"best cost (reported) : {best_cost:.6g}")
    print(f"best cost (re-eval)  : {final_cost:.6g}")
    print(f"layout shape         : {best_layout.shape}")
    print("=" * 70)
    print()


def main():
    print("\nHILL CLIMBER - MULTI-SCENARIO RUN\n")

    # without obstacles
    run_scenario_hill_climb("../Scenarios/01.xml")
    #with obstacles
    run_scenario_hill_climb("../Scenarios/obs_05.xml")

    print("\nGENETIC ALGORITHM - MULTI-SCENARIO RUN\n")
    # without obstacles
    run_scenario_ga("../Scenarios/01.xml")
    #with obstacles
    run_scenario_ga("../Scenarios/obs_05.xml")





if __name__ == "__main__":
    main()