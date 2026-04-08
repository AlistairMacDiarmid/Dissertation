"""
maintain a population of different layouts and evolve them over time by combining good solutions and adding random changes.

create a population of random valid layouts (first gen)
evaluate how good each layout is
select the best layout as the parents
create children by mixing the parents together
randomly tweak some children
keep the best layouts and discard the worst
repeat for many generations
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from KusiakEnergyEvaluator import KusiakEnergyEvaluator
from wind_scenario import WindScenario
from generate_layout import generate_initial_feasible_layout, point_in_any_obstacle



@dataclass
class GAResult:
    """
    data class to store the outcome of one GA run
    """
    best_layout: np.ndarray
    best_cost: float
    eval_history: list[int]
    best_cost_history: list[float]
    mean_cost_history: list[float]
    total_evals: int
    runtime: float

class EvalCounter:
    """
    wrapper for the evaluator to count how many objective-function evals have been performed
    """

    def __init__(self, scenario: WindScenario):
        self.evaluator = KusiakEnergyEvaluator(scenario)
        self.eval_count = 0

    def evaluate(self, layout: np.ndarray) -> float:
        self.eval_count += 1
        return float(self.evaluator.evaluate(layout))


def is_valid_turbine_move(
        layout: np.ndarray,
        turbine_index: int,
        proposed_x: float,
        proposed_y: float,
        scenario: WindScenario,
) -> bool:
    """
    quick check to see if the turbine can move to a new spot
    :param layout: current positions of all turbines
    :param turbine_index: which turbine is being checked
    :param proposed_x: new x pos to test
    :param proposed_y: new y pos to test
    :param scenario: all farm constraints
    :return: return True if the pos is valid, False otherwise
    """

    #check if pos is outside farm bounds
    if proposed_x < 0.0 or proposed_y < 0.0:
        return False
    if proposed_x > float(scenario.width) or proposed_y > float(scenario.height):
        return False

    #check if the pos is inside an obstacle
    if point_in_any_obstacle(proposed_x, proposed_y, scenario.obstacles):
        return False

    #check if the pos is too close to any other turbines
    dx = layout[:, 0] - proposed_x
    dy = layout[:, 1] - proposed_y
    #ignore the distance to itself
    dx[turbine_index] = np.inf
    dy[turbine_index] = np.inf
    distance_squared =dx * dx + dy * dy
    return not bool(np.any(distance_squared < float(scenario.minDist)))


def tournament_select(
        costs: np.ndarray,
        rng: np.random.Generator,
        tournament_size: int = 3,
)-> int:
    """
    return the index of the best individual among random tournament contenders
    :param costs: energy costs of all layouts in the population
    :param rng: random number generator
    :param tournament_size: how many layouts to compete in each tournament
    :return: index of winning layout
    """
    population_size = costs.shape[0]
    #randomly pick some layouts to compete
    contenders = rng.integers(0, population_size, size=tournament_size)
    #lowest cost wins
    best_index = contenders[np.argmin(costs[contenders])]
    return best_index


def crossover_positions(
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        scenario: WindScenario,
        rng: np.random.Generator,
        num_imports: int = 10,
) -> np.ndarray:
    """
    create a child layout from mixing the two parents layouts

    :param parent_a: first parent layout (the base for the child)
    :param parent_b: second parent layout (the source of new positions)
    :param scenario: all farm constraints
    :param rng: random number generator
    :param num_imports: how many positions are to be attempted to be imported from parent b
    :return: return child layout
    """

    #copy parent A
    child = parent_a.copy()
    nturb = child.shape[0]

    #import multiple turbine positions from parent B
    for i in range(num_imports):
        #select a random turbine pos to import
        turbine_index = int(rng.integers(0, nturb))
        proposed_x, proposed_y = parent_b[turbine_index]

        #only use the pos if its valid in the child layout
        if is_valid_turbine_move(child, turbine_index, float(proposed_x), float(proposed_y), scenario):
            child[turbine_index] = (proposed_x, proposed_y)
            #if the pos is not valid, skip and keep pos from parent A


    return child


def mutate(
        layout: np.ndarray,
        scenario: WindScenario,
        rng: np.random.Generator,
        step_size: float = 50.0,
        mutations: int = 3,
) -> np.ndarray:
    """
    randomly tweak a layout
    :param layout: layout to be mutated
    :param scenario: all farm constraints
    :param rng: random number generator
    :param step_size: how far (meters) a turbine can move
    :param mutations: how many turbines to attempt to mutate
    :return: return the mutated layout
    """

    mutant = layout.copy()
    nturb = layout.shape[0]

    #try to mutate multiple randomly selected turbines
    for i in range(mutations):
        #select random turbine
        turbine_index = int(rng.integers(0, nturb))
        original_x, original_y = mutant[turbine_index]

        #nudge randomly in both x and y directions
        proposed_x = float(original_x + rng.uniform(-step_size, step_size))
        proposed_y = float(original_y + rng.uniform(-step_size, step_size))

        #apply the mutation only if the new pos is valid
        if is_valid_turbine_move(mutant, turbine_index, float(proposed_x), float(proposed_y), scenario):
            mutant[turbine_index] = (proposed_x, proposed_y)
            #turbine stays where it is if not valid.

    return mutant



def genetic_algorithm(
        scenario: WindScenario,
        *,
        population_size: int = 10,
        generations: int = 20,
        elite_count: int = 2,
        tournament_size: int = 3,
        step_size: float = 50.0,
        crossover_imports: int = 10,
        mutation_count: int = 10,
        log_every: int = 1,
        seed: int | None = None,
        record_mean: bool = True,
) -> GAResult:
    """
    evolve a population of layouts over a defined number of generations

    the convergence is tracked by KusiakEEnergyEvaluator calls

    """
    rng = np.random.default_rng(seed)
    evaluator = EvalCounter(scenario)

    start_time = time.perf_counter()

    # generate the initial population
    population: list[np.ndarray] = []
    for _ in range(population_size):
        individual_seed = int(rng.integers(0, 2 ** 32 - 1))
        individual = generate_initial_feasible_layout(scenario, seed=individual_seed)
        population.append(individual)

    population = np.array(population, dtype=float)

    # history tracked via eval calls
    eval_history: list[int] = []
    best_cost_history: list[float] = []
    mean_cost_history: list[float] | None = [] if record_mean else None

    global_best_cost = float("inf")
    global_best_layout: np.ndarray | None = None

    # evaluate the initial population
    cost_list: list[float] = []
    for individual in population:
        cost = evaluator.evaluate(individual)
        cost_list.append(cost)

        if cost < global_best_cost:
            global_best_cost = cost
            global_best_layout = individual.copy()

        eval_history.append(evaluator.eval_count)
        best_cost_history.append(global_best_cost)

    costs = np.array(cost_list, dtype=float)

    # the main GA loop
    for generation in range(1, generations + 1):
        # sort in ascending order
        order = np.argsort(costs)
        population = population[order]
        costs = costs[order]

        generation_best_cost = float(costs[0])
        generation_mean_cost = float(np.mean(costs))

        if mean_cost_history is not None:
            mean_cost_history.append(generation_mean_cost)

        if log_every and (generation % log_every == 0):
            print(
                f"GA generation={generation:3d}  "
                f"evals={evaluator.eval_count:5d}  "
                f"best(gen)={generation_best_cost:.9f}  "
                f"best(global)={global_best_cost:.9f}  "
                f"mean={generation_mean_cost:.9f}"
            )

        # make the next generation
        new_population = [population[i].copy() for i in range(elite_count)]
        new_costs = [float(costs[i]) for i in range(elite_count)]

        while len(new_population) < population_size:
            parent_one_index = tournament_select(costs, rng, tournament_size)
            parent_two_index = tournament_select(costs, rng, tournament_size)

            parent_a = population[parent_one_index]
            parent_b = population[parent_two_index]

            child = crossover_positions(
                parent_a,
                parent_b,
                scenario,
                rng,
                num_imports=crossover_imports,
            )

            child = mutate(
                child,
                scenario,
                rng,
                step_size=step_size,
                mutations=mutation_count,
            )

            child_cost = evaluator.evaluate(child)

            # fallback if evaluator returns invalid
            if not np.isfinite(child_cost):
                fallback_seed = int(rng.integers(0, 2 ** 32 - 1))
                child = generate_initial_feasible_layout(scenario, seed=fallback_seed)
                child_cost = evaluator.evaluate(child)

            new_population.append(child)
            new_costs.append(child_cost)

            if child_cost < global_best_cost:
                global_best_cost = child_cost
                global_best_layout = child.copy()

            eval_history.append(evaluator.eval_count)
            best_cost_history.append(global_best_cost)

        population = np.array(new_population, dtype=float)
        costs = np.array(new_costs, dtype=float)

    # final results
    if global_best_layout is None:
        best_index = int(np.argmin(costs))
        global_best_layout = population[best_index].copy()
        global_best_cost = float(costs[best_index])

    runtime = time.perf_counter() - start_time

    return GAResult(
        best_layout=global_best_layout,
        best_cost=global_best_cost,
        eval_history=eval_history,
        best_cost_history=best_cost_history,
        mean_cost_history=mean_cost_history,
        total_evals=evaluator.eval_count,
        runtime=runtime,
    )

# def genetic_algorithm(
#         scenario: WindScenario,
#         *,
#         population_size: int = 10,
#         generations: int = 20,
#         elite_count: int = 2,
#         tournament_size: int = 3,
#         step_size: float = 50.0,
#         crossover_imports: int = 10,
#         mutation_count: int = 10,
#         log_every: int = 1,
#         seed: int | None = None,
#         record_mean: bool = True,
# ) -> tuple[np.ndarray, float, list[float], list[float] | None]:
#     """
#     Evolve a population of layouts over multiple generations.
#
#     Returns:
#         best_layout: best layout found
#         best_cost: best (lowest) energy cost found
#         best_cost_history: global-best cost after each generation (for convergence plots)
#         mean_cost_history: mean cost each generation (optional; useful for diagnostics)
#     """
#
#     rng = np.random.default_rng(seed)
#     evaluator = KusiakEnergyEvaluator(scenario)
#
#     eval_count: int = 0 # track the evaluator calls
#     cost_history: list[float] = []
#     eval_history: list[float] = []
#
#     # ---------------------------
#     # 1) Initial population
#     # ---------------------------
#     population: list[np.ndarray] = []
#     for _ in range(population_size):
#         individual_seed = int(rng.integers(0, 2**32 - 1))
#         individual = generate_initial_feasible_layout(scenario, seed=individual_seed)
#         population.append(individual)
#
#     population = np.array(population, dtype=float)
#
#     # Evaluate initial population
#     costs = np.array([float(evaluator.evaluate(individual)) for individual in population], dtype=float)
#
#     # Convergence history
#     best_cost_history: list[float] = []
#     mean_cost_history: list[float] | None = [] if record_mean else None
#
#     # Track global best-so-far (this makes the convergence curve nicer + monotonic)
#     global_best_cost = float("inf")
#     global_best_layout: np.ndarray | None = None
#
#     # ---------------------------
#     # 2) Main GA loop
#     # ---------------------------
#     for generation_index in range(1, generations + 1):
#         # Sort by cost (ascending)
#         order = np.argsort(costs)
#         population = population[order]
#         costs = costs[order]
#
#         generation_best_cost = float(costs[0])
#         generation_mean_cost = float(np.mean(costs))
#
#         # Update global best
#         if generation_best_cost < global_best_cost:
#             global_best_cost = generation_best_cost
#             global_best_layout = population[0].copy()
#
#         # Record history (global best curve is what you want for convergence comparison)
#         best_cost_history.append(global_best_cost)
#         if mean_cost_history is not None:
#             mean_cost_history.append(generation_mean_cost)
#
#         # Logging
#         if log_every and (generation_index % log_every == 0):
#             print(
#                 f"GA gen={generation_index:3d}  "
#                 f"best(gen)={generation_best_cost:.6g}  "
#                 f"best(global)={global_best_cost:.6g}  "
#                 f"mean={generation_mean_cost:.6g}"
#             )
#
#         # ---------------------------
#         # 3) Create next generation
#         # ---------------------------
#         new_population = [population[i].copy() for i in range(elite_count)]
#
#         while len(new_population) < population_size:
#             p1_idx = tournament_select(costs, rng, tournament_size)
#             p2_idx = tournament_select(costs, rng, tournament_size)
#
#             parent_a = population[p1_idx]
#             parent_b = population[p2_idx]
#
#             child = crossover_positions(parent_a, parent_b, scenario, rng, num_imports=crossover_imports)
#             child = mutate(child, scenario, rng, step_size=step_size, mutations=mutation_count)
#
#             child_cost = float(evaluator.evaluate(child))
#             if not np.isfinite(child_cost):
#                 fallback_seed = int(rng.integers(0, 2 ** 32 - 1))
#                 child = generate_initial_feasible_layout(scenario, seed=fallback_seed)
#
#             new_population.append(child)
#
#         population = np.array(new_population, dtype=float)
#         costs = np.array([float(evaluator.evaluate(individual)) for individual in population], dtype=float)
#
#     # ---------------------------
#     # 4) Final best
#     # ---------------------------
#     if global_best_layout is None:
#         # Very defensive fallback; should never happen if population_size > 0
#         best_index = int(np.argmin(costs))
#         global_best_layout = population[best_index].copy()
#         global_best_cost = float(costs[best_index])
#
#     return global_best_layout, global_best_cost, best_cost_history, mean_cost_history


# def genetic_algorithm(
#         scenario: WindScenario,
#         *,
#         population_size: int = 10,
#         generations: int = 20,
#         elite_count: int = 2,
#         tournament_size: int = 3,
#         step_size: float = 50.0,
#         crossover_imports: int = 10,
#         mutation_count: int = 10,
#         log_every: int = 1,
#         seed: int | None = None
# ) -> tuple[np.ndarray, float]:
#     """
#     evolve a population of layouts over multiple generations, gradually improving them through selection, crossover and mutation
#     :param seed:
#     :param scenario: farm constraints
#     :param population_size: how many layouts to maintain each generation
#     :param generations: how many generations to evolve
#     :param elite_count: how many best layouts to automatically keep each generation
#     :param tournament_size: how many layouts to compete in each selection tournament
#     :param step_size: how for mutations can move each turbine (meters)
#     :param crossover_imports: how many positions to try importing during the crossover
#     :param mutation_count: how many turbines to attempt mutating per child
#     :param log_every: print progress every N generations
#     :return: returns a tuple of (best_layout, best_energy_cost)
#     """
#
#     rng = np.random.default_rng(seed)
#     evaluator = KusiakEnergyEvaluator(scenario)
#
#     #create the initial population
#     population = []
#     for _ in range(population_size):
#         individual_seed = int(rng.integers(0,2**32-1))
#         individual = generate_initial_feasible_layout(scenario, seed=individual_seed)
#         population.append(individual)
#
#     population = np.array(population, dtype=float)
#
#     #evaluate how good each layout is
#     costs = np.array([float(evaluator.evaluate(individual)) for individual in population], dtype=float)
#
#     #main loop
#     for generation_index in range(1, generations + 1):
#         #sort population by fitness
#         order = np.argsort(costs)
#         population = population[order]
#         costs = costs[order]
#
#         best_cost = float(costs[0])
#
#         #print progress
#         if log_every and (generation_index % log_every == 0):
#             print(
#                 f"Genetic Algorithm: generation={generation_index:3d}, best cost={best_cost:.6g}, mean cost = {float(np.mean(costs)):.6g}"
#             )
#
#         #create next generation
#
#         #automatically keep the best layouts
#         new_population = [population[i].copy() for i in range(elite_count)]
#
#         #fill the rest of the population with new children
#         while len(new_population) < population_size:
#             #select two parents using tournament selection
#             p1_idx = tournament_select(costs, rng, tournament_size)
#             p2_idx = tournament_select(costs, rng, tournament_size)
#
#             parent_a = population[p1_idx]
#             parent_b = population[p2_idx]
#
#             #create the child by combining the two parents
#             child = crossover_positions(parent_a, parent_b, scenario, rng, num_imports=crossover_imports)
#
#             #mutate the child to add variety
#             child = mutate(child, scenario, rng, step_size=step_size, mutations=mutation_count)
#
#             #check if something went wrong and the child is invalid, if so generate a fresh valid layout
#             child_cost = float(evaluator.evaluate(child))
#             if not np.isfinite(child_cost):
#                 fallback_seed = int(rng.integers(0, 2 ** 32 - 1))
#                 child = generate_initial_feasible_layout(scenario, seed=fallback_seed)
#
#             new_population.append(child)
#
#         # children become new population for next generation
#         population = np.array(new_population, dtype=float)
#         costs = np.array(
#             [float(evaluator.evaluate(individual)) for individual in population],
#             dtype=float,
#         )
#
#     #return the bests across all generations
#     best_index = int(np.argmin(costs))
#     best_layout = population[best_index]
#     best_cost = float(costs[best_index])
#
#     return best_layout, best_cost


