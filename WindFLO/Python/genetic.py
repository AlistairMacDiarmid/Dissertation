"""
maintain a population of different layouts and evolve them over time by combining good solutions and adding random changes.

- initialise a population of feasible layouts
- evaluate fitness using the energy cost function
- select parents using tournament selection
- generate offspring via crossover and mutation
- retain the best individuals (elitism)
- repeat for multiple generations
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from KusiakEnergyEvaluator import KusiakEnergyEvaluator
from wind_scenario import WindScenario
from generate_layout import generate_initial_feasible_layout, point_in_any_obstacle

#This module implements a population-based optimisation approach
# (genetic algorithm) for the wind farm layout optimisation problem.
#
# Unlike hill-climbing, which follows a single trajectory, the GA
# maintains multiple candidate solutions and explores the search space
# through recombination and stochastic variation.


@dataclass
class GAResult:
    """
    data class to store the outcome of one GA run
    Attributes:

    best_layout: best turbine layout found
    best_cost: corresponding objective value
    eval_history: evaluator calls over time
    best_cost_history: global best cost after each evaluation
    mean_cost_history: average population cost per generation
    total_evals: total number of evaluations performed
    runtime: execution time in seconds

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

    this is used for:
    - reporting computational effort
    - comparing algorithms fairly using evaluator calls
    - plotting convergence against evaluation count
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

    Ensures:
    - turbine remains within farm boundaries
    - turbine is not placed inside obstacles
    - minimum spacing constraint is satisfied

    :param layout: current positions of all turbines
    :param turbine_index: which turbine is being checked
    :param proposed_x: new x pos to test
    :param proposed_y: new y pos to test
    :param scenario: all farm constraints
    :return: return True if the pos is valid, False otherwise
    """

    #enforce boundary constraints
    if proposed_x < 0.0 or proposed_y < 0.0:
        return False
    if proposed_x > float(scenario.width) or proposed_y > float(scenario.height):
        return False

    #enforce obstacle constraints
    if point_in_any_obstacle(proposed_x, proposed_y, scenario.obstacles):
        return False

    #enforce minimum spacing constraint
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
    Select a parent using tournament selection.

    A subset of individuals is randomly sampled, and the individual
    with the lowest cost is selected.

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
    Perform turbine-level crossover between two parent layouts.

    Instead of swapping full layouts, this operator selectively imports
    turbine positions from parent B into parent A.

    :param parent_a: first parent layout (the base for the child)
    :param parent_b: second parent layout (the source of new positions)
    :param scenario: all farm constraints
    :param rng: random number generator
    :param num_imports: how many positions are to be attempted to be imported from parent b
    :return: return child layout
    """


    child = parent_a.copy()
    nturb = child.shape[0]

    for i in range(num_imports):
        turbine_index = int(rng.integers(0, nturb))
        proposed_x, proposed_y = parent_b[turbine_index]

        #apply only is feasible
        if is_valid_turbine_move(child, turbine_index, float(proposed_x), float(proposed_y), scenario):
            child[turbine_index] = (proposed_x, proposed_y)

    return child


def mutate(
        layout: np.ndarray,
        scenario: WindScenario,
        rng: np.random.Generator,
        step_size: float = 50.0,
        mutations: int = 3,
) -> np.ndarray:
    """
    Apply mutation by perturbing turbine positions

    Each mutation:
    - selects a random turbine
    - applies a bounded displacement
    - is accepted only if feasible

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
        # sort in ascending fitness order
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

        # elitism to preserve the best individuals
        new_population = [population[i].copy() for i in range(elite_count)]
        new_costs = [float(costs[i]) for i in range(elite_count)]

        #generate offspring
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

            #update the global best
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