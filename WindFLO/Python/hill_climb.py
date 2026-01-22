from __future__ import annotations
import numpy as np

from KusiakEnergyEvaluator import KusiakEnergyEvaluator
from wind_scenario import WindScenario
from generate_layout import generate_initial_feasible_layout, point_in_any_obstacle

def _is_valid_turbine_move(
        layout: np.ndarray,
        turb_idx: int,
        new_x: float,
        new_y: float,
        scenario: WindScenario,
) -> bool:
    """
    this is a fast feasibility check for moving just ONE turbine.
    this will avoid calling evaluator.checkConstraint(layout), which is O(n^2).
    it instead will perform an O(n) check by only testing the moving turbine against:
        - farm boundary constraints
        - obstacle avoidance
        - minimum spacing against all other turbines
    :param layout: Current layout array (n x 2)
    :param turb_idx: index of the turbine being moved
    :param new_x: proposed new x coordinate
    :param new_y: proposed new y coordinate
    :param scenario: WindScenario with constraints
    :return: True if the move is valid, False otherwise
    """

    #check if the new position is within the farm bounds
    if new_x < 0.0 or new_y < 0.0 or new_x > float(scenario.width) or new_y > float(scenario.height):
        return False

    #check if the new position is inside any obstacle
    if point_in_any_obstacle(new_x, new_y, scenario.obstacles):
        return False

    #check the spacing against all other turbines and calculate the distance from a new position to all other turbines
    dx = layout[:, 0] - new_x
    dy = layout[:, 1] - new_y

    #ignore the distance to itself by setting to infinity. To prevent false rejection when checking turbine against its own position
    dx[turb_idx] = np.inf
    dy[turb_idx] = np.inf

    #check if any turbine is too close
    dist_sq = dx * dx + dy * dy

    return not bool(np.any(dist_sq < float(scenario.minDist)))

def hill_climb(
        scenario: WindScenario,
        iterations: int = 50,
        step_size: float = 50.0,
        max_attempts_per_iteration: int = 20,
        log_every: int = 10,
):
    """
    start from an initial feasible layout
    repeatedly apply small random perturbations
    accept the first improving feasible neighbour

    :param scenario:
    :param iterations:
    :param step_size:
    :param max_attempts_per_iteration:
    :param log_every:
    :return:
    """

    random_generator = np.random.default_rng()
    evaluator = KusiakEnergyEvaluator(scenario)

    #generate the initial feasible layout solution
    current_layout_positions = generate_initial_feasible_layout(scenario)
    current_energy_cost = float(evaluator.evaluate(current_layout_positions))

    best_layout_positions = current_layout_positions.copy()
    best_energy_cost = current_energy_cost

    number_of_turbines = int(scenario.nturbines)

    for iteration_index in range(1, iterations + 1):
        improvement_found = False

        #try multiple local perturbations per iteration
        for proposal_attempt in range(max_attempts_per_iteration):
            turbine_index = int(random_generator.integers(0, number_of_turbines))

            original_x, original_y = current_layout_positions[turbine_index]

            proposed_x = original_x + float(random_generator.uniform(-step_size, step_size))
            proposed_y = original_y + float(random_generator.uniform(-step_size, step_size))

            #feasibility check before evaluation
            if not _is_valid_turbine_move(
                current_layout_positions,
                turbine_index,
                proposed_x,
                proposed_y,
                scenario,
            ):
                continue

            #apply the move temporarily
            current_layout_positions[turbine_index] = (proposed_x, proposed_y)
            candidate_energy_cost = float(evaluator.evaluate(current_layout_positions))

            if candidate_energy_cost < current_energy_cost:
                current_energy_cost = candidate_energy_cost
                improvement_found = True

                if candidate_energy_cost < best_energy_cost:
                    best_energy_cost = candidate_energy_cost
                    best_layout_positions = current_layout_positions.copy()
                break #accept the move and go onto the next iteration
            else:
                current_layout_positions[turbine_index] = (original_x, original_y) #revert proposed move

        # Progress logging
        if log_every and (iteration_index % log_every == 0):
            status = "accepted move" if improvement_found else "no improvement"
            print(
                f"hill-climb iteration={iteration_index:4d}  "
                f"best_cost={best_energy_cost:.6g}  "
                f"current_cost={current_energy_cost:.6g}  "
                f"status={status}"
            )

    return best_layout_positions, best_energy_cost
