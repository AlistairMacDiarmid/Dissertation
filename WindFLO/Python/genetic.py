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

from KusiakEnergyEvaluator import KusiakEnergyEvaluator
from wind_scenario import WindScenario
from generate_layout import generate_initial_feasible_layout, point_in_any_obstacle

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
            #turbine stays where it is if its not valid.

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
) -> tuple[np.ndarray, float]:
    """
    evolve a population of layouts over multiple generations, gradually improving them through selection, crossover and mutation
    :param scenario: farm constraints
    :param population_size: how many layouts to maintain each generation
    :param generations: how many generations to evolve
    :param elite_count: how many best layouts to automatically keep each generation
    :param tournament_size: how many layouts to compete in each selection tournament
    :param step_size: how for mutations can move each turbine (meters)
    :param crossover_imports: how many positions to try importing during the crossover
    :param mutation_count: how many turbines to attempt mutating per child
    :param log_every: print progress every N generations
    :return: returns a tuple of (best_layout, best_energy_cost)
    """

    rng = np.random.default_rng()
    evaluator = KusiakEnergyEvaluator(scenario)

    #create the initial population
    population = []
    for _ in range(population_size):
        individual = generate_initial_feasible_layout(scenario)
        population.append(individual)

    population = np.array(population, dtype=float)

    #evaluate how good each layout is
    costs = np.array([float(evaluator.evaluate(individual)) for individual in population], dtype=float)

    #main loop
    for generation_index in range(1, generations + 1):
        #sort population by fitness
        order = np.argsort(costs)
        population = population[order]
        costs = costs[order]

        best_cost = float(costs[0])

        #print progress
        if log_every and (generation_index % log_every == 0):
            print(
                f"Genetic Algorithm: generation={generation_index:3d}, best cost={best_cost:.6g}, mean cost = {float(np.mean(costs)):.6g}"
            )

        #create next generation

        #automatically keep the best layouts
        new_population = [population[i].copy() for i in range(elite_count)]

        #fill the rest of the population with new children
        while len(new_population) < population_size:
            #select two parents using tournament selection
            p1_idx = tournament_select(costs, rng, tournament_size)
            p2_idx = tournament_select(costs, rng, tournament_size)

            parent_a = population[p1_idx]
            parent_b = population[p2_idx]

            #create the child by combining the two parents
            child = crossover_positions(parent_a, parent_b, scenario, rng, num_imports=crossover_imports)

            #mutate the child to add variety
            child = mutate(child, scenario, rng, step_size=step_size, mutations=mutation_count)

            #check if something went wrong and the child is invalid, if so generate a fresh valid layout
            child_cost = float(evaluator.evaluate(child))
            if not np.isfinite(child_cost):
                child = generate_initial_feasible_layout(scenario)
                child_cost = float(evaluator.evaluate(child))

            new_population.append(child)

        # children become new population for next generation
        population = np.array(new_population, dtype=float)
        costs = np.array(
            [float(evaluator.evaluate(individual)) for individual in population],
            dtype=float,
        )

    #return the best layout across all generations
    best_index = int(np.argmin(costs))
    return population[best_index], costs[best_index]



