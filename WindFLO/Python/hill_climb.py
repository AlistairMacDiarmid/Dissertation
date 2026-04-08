from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np

from KusiakEnergyEvaluator import KusiakEnergyEvaluator
from wind_scenario import WindScenario
from generate_layout import generate_initial_feasible_layout, point_in_any_obstacle

# This module implements three local-search optimisation methods
# for the wind farm layout optimisation problem:
#
# 1. basic hill climb
# 2. stochastic hill climb
# 3. random-restart hill climb
#
# All methods start from a feasible layout and attempt to reduce
# the objective value returned by the KusiakEnergyEvaluator.
#
# The core idea is to perturb one turbine at a time, check whether
# the move remains feasible, and only evaluate layouts that satisfy
# the scenario constraints.


@dataclass
class HCResult:
    """
    result for one hill-climber run


    Attributes:
    best_layout:
        Best turbine layout found during the run.

    best_cost:
        Objective value of the best layout.

    eval_history:
        Cumulative evaluator-call counts recorded during the run.

    best_cost_history:
        Best-so-far objective value after each real evaluation.

    total_evals:
        Total number of objective-function calls made.

    runtime:
        Total wall-clock runtime in seconds.
    """
    best_layout: np.ndarray
    best_cost: float
    eval_history: list[int]
    best_cost_history: list[float]
    total_evals: int
    runtime: float


class EvalCounter:
    """
    wrapper for evaluator so objective-function calls can be counted.

    this is used for:
    - reporting computational effort
    - comparing algorithms fairly using evaluator calls
    - plotting convergence against evaluation count
    """

    def __init__(self, scenario: WindScenario):
        self.evaluator = KusiakEnergyEvaluator(scenario)
        self.eval_count = 0

    def evaluate(self, layout: np.ndarray) -> float:
        #evaluate the layout and increment the objective-function count
        self.eval_count += 1
        return float(self.evaluator.evaluate(layout))


def _is_valid_turbine_move(
        layout: np.ndarray,
        turb_idx: int,
        new_x: float,
        new_y: float,
        scenario: WindScenario,
) -> bool:
    """
    fast feasibility check for moving one turbine.

    Instead of validating the entire layout from scratch, this checks only whether the proposed turbine position:
    - remains inside the farm boundary
    - does not fall inside an obstacle
    - does not violate the minimum spacing constraint
    """

    #reject move if turbine exceeds the farms boundaries
    if new_x < 0.0 or new_y < 0.0 or new_x > float(scenario.width) or new_y > float(scenario.height):
        return False

    #reject move if turbine is in any obstacle region
    if point_in_any_obstacle(new_x, new_y, scenario.obstacles):
        return False

    #compute the distance from proposed turbine pos to all turbines
    dx = layout[:, 0] - new_x
    dy = layout[:, 1] - new_y

    #ignore the turbine being moved
    dx[turb_idx] = np.inf
    dy[turb_idx] = np.inf

    #spacing enforced using squared distance to avoid sqrt calls
    dist_sq = dx * dx + dy * dy
    return not bool(np.any(dist_sq < float(scenario.minDist)))


def hill_climb(
        scenario: WindScenario,
        iterations: int = 250,
        step_size: float = 50.0,
        max_attempts_per_iteration: int = 25,
        log_every: int = 10,
        seed: int | None = None,
) -> HCResult:
    """
    basic hill climber:
    - start from a feasible initial layout
    - repeatedly generate feasible neighbours by perturbing one turbine
    - accept the first improving move found
    - reject non-improving moves

    """

    rng = np.random.default_rng(seed)
    evaluator = EvalCounter(scenario)

    start_time = time.perf_counter()

    #generate initial layout which satisfies all constraints
    current_layout = generate_initial_feasible_layout(scenario, seed=seed)
    current_cost = evaluator.evaluate(current_layout)

    #initislise best-so-far tracking
    best_layout = current_layout.copy()
    best_cost = current_cost

    #start convergence history with the initial evaluation
    eval_history: list[int] = [evaluator.eval_count]
    best_cost_history: list[float] = [best_cost]

    nturbines = int(scenario.nturbines)

    #the main optimisation loop
    for iteration_index in range(1, iterations + 1):
        improvement_found = False

        #allow multiple neighbour attempts per iteration
        for _ in range(max_attempts_per_iteration):
            turbine_index = int(rng.integers(0, nturbines))

            original_x, original_y = current_layout[turbine_index]

            #propose a local perturbation bounded by the step size
            proposed_x = float(original_x + rng.uniform(-step_size, step_size))
            proposed_y = float(original_y + rng.uniform(-step_size, step_size))

            #skip if turbine move is infeasible
            if not _is_valid_turbine_move(
                current_layout,
                turbine_index,
                proposed_x,
                proposed_y,
                scenario,
            ):
                continue

            #temporarily apply move so candidate layout can be evaluated
            current_layout[turbine_index] = (proposed_x, proposed_y)
            candidate_cost = evaluator.evaluate(current_layout)

            #greedy acceptance, only accept improving moves
            if candidate_cost < current_cost:
                current_cost = candidate_cost
                improvement_found = True

                #update best-so-far record if its the best layout seen
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_layout = current_layout.copy()
            else:
                #revert move if it did not improve the objective
                current_layout[turbine_index] = (original_x, original_y)

            # record convergence after every real evaluation call
            eval_history.append(evaluator.eval_count)
            best_cost_history.append(best_cost)

            #stop searching for neighbours once an improving move has been accepted
            if improvement_found:
                break

        if log_every and (iteration_index % log_every == 0):
            status = "accepted move" if improvement_found else "no improvement"
            print(
                f"hill-climb iteration={iteration_index:4d}  "
                f"evals={evaluator.eval_count:5d}  "
                f"best_cost={best_cost:.9f}  "
                f"current_cost={current_cost:.9f}  "
                f"status={status}"
            )

    runtime = time.perf_counter() - start_time

    return HCResult(
        best_layout=best_layout,
        best_cost=best_cost,
        eval_history=eval_history,
        best_cost_history=best_cost_history,
        total_evals=evaluator.eval_count,
        runtime=runtime,
    )


def stochastic_hill_climb(
        scenario: WindScenario,
        iterations: int = 250,
        step_size: float = 50.0,
        log_every: int = 10,
        seed: int | None = None,
) -> HCResult:
    """
    stochastic hill climber:
    - generate exactly one random neighbour per iteration
    - accept it only if it improves the objective

    Compared with the basic hill climber, this samples the
    neighbourhood more loosely and may explore more diverse moves,
    """

    rng = np.random.default_rng(seed)
    evaluator = EvalCounter(scenario)

    start_time = time.perf_counter()

    current_layout = generate_initial_feasible_layout(scenario, seed=seed)
    current_cost = evaluator.evaluate(current_layout)

    best_layout = current_layout.copy()
    best_cost = current_cost

    eval_history: list[int] = [evaluator.eval_count]
    best_cost_history: list[float] = [best_cost]

    nturbines = int(scenario.nturbines)

    for iteration_index in range(1, iterations + 1):
        turbine_index = int(rng.integers(0, nturbines))

        original_x, original_y = current_layout[turbine_index]
        proposed_x = float(original_x + rng.uniform(-step_size, step_size))
        proposed_y = float(original_y + rng.uniform(-step_size, step_size))

        #evaluate only if the random neighbour is feasible
        if _is_valid_turbine_move(current_layout, turbine_index, proposed_x, proposed_y, scenario):
            current_layout[turbine_index] = (proposed_x, proposed_y)
            candidate_cost = evaluator.evaluate(current_layout)

            if candidate_cost < current_cost:
                current_cost = candidate_cost

                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_layout = current_layout.copy()
            else:
                #revert non-improving moves
                current_layout[turbine_index] = (original_x, original_y)

            # record convergence after every real evaluation call
            eval_history.append(evaluator.eval_count)
            best_cost_history.append(best_cost)

        if log_every and (iteration_index % log_every == 0):
            print(
                f"stochastic hill climb iteration={iteration_index:4d}  "
                f"evals={evaluator.eval_count:5d}  "
                f"best_cost={best_cost:.9f}  "
                f"current_cost={current_cost:.9f}"
            )

    runtime = time.perf_counter() - start_time

    return HCResult(
        best_layout=best_layout,
        best_cost=best_cost,
        eval_history=eval_history,
        best_cost_history=best_cost_history,
        total_evals=evaluator.eval_count,
        runtime=runtime,
    )


def hill_climb_random_restarts(
        scenario: WindScenario,
        restarts: int = 15,
        iterations: int = 250,
        step_size: float = 50.0,
        max_attempts_per_iteration: int = 25,
        seed: int | None = None,
        log_every: int = 1,
) -> HCResult:
    """
    random-restart hill climber:
    - run the basic hill climber multiple times
    - use a different random seed each time
    - keep the best result found across all restarts

    This improves robustness by sampling multiple starting points in
    the search space
    """

    print("\n" + "-" * 50)
    print("RR-HC (Random-Restart Hill Climber)")
    print("-" * 50)
    print(f"restarts                 : {restarts}")
    print(f"iterations / restart     : {iterations}")
    print(f"step_size                : {step_size}")
    print(f"max_attempts / iteration : {max_attempts_per_iteration}")
    print(f"master seed              : {seed}")
    print("-" * 50)

    rng = np.random.default_rng(seed)

    #best solution found across all restarts
    global_best_layout: np.ndarray | None = None
    global_best_cost = float("inf")

    #global convergence history merged onto one continuous evaluation axis
    global_eval_history: list[int] = []
    global_best_cost_history: list[float] = []

    total_evals_so_far = 0

    total_start = time.perf_counter()

    for restart_index in range(1, restarts + 1):
        #draw a fresh seed for this restart so each run begins differently
        restart_seed = int(rng.integers(0, 2**31 - 1))
        restart_start = time.perf_counter()

        result = hill_climb(
            scenario=scenario,
            iterations=iterations,
            step_size=step_size,
            max_attempts_per_iteration=max_attempts_per_iteration,
            seed=restart_seed,
            log_every=0,
        )

        restart_time = time.perf_counter() - restart_start

        #update the global best solution if restart improved on it
        if result.best_cost < global_best_cost:
            global_best_cost = result.best_cost
            global_best_layout = result.best_layout.copy()

        # merge restart history into one continuous evaluation axis
        for restart_eval, restart_best in zip(result.eval_history, result.best_cost_history):
            absolute_eval = total_evals_so_far + restart_eval

            #maintain a running global best across all completed restarts
            if global_best_cost_history:
                running_best = min(global_best_cost_history[-1], restart_best)
            else:
                running_best = restart_best

            global_eval_history.append(absolute_eval)
            global_best_cost_history.append(running_best)

        total_evals_so_far += result.total_evals

        if log_every and (restart_index % log_every == 0):
            print(
                f"[RR-HC] restart={restart_index:3d}/{restarts}  "
                f"time={restart_time:6.2f}s  "
                f"seed={restart_seed:<10d}  "
                f"restart_best={result.best_cost:.9f}  "
                f"global_best={global_best_cost:.9f}  "
                f"evals_so_far={total_evals_so_far}"
            )

    total_runtime = time.perf_counter() - total_start

    if global_best_layout is None:
        raise RuntimeError("random-restart hill climber failed to produce a valid result")

    print("\n" + "-" * 50)
    print("RR-HC SUMMARY")
    print("-" * 50)
    print(f"total runtime (s)        : {total_runtime:.2f}")
    print(f"avg per restart (s)      : {total_runtime / restarts:.2f}")
    print(f"best overall cost        : {global_best_cost:.9f}")
    print(f"total evaluator calls    : {total_evals_so_far}")
    print("-" * 50 + "\n")

    return HCResult(
        best_layout=global_best_layout,
        best_cost=float(global_best_cost),
        eval_history=global_eval_history,
        best_cost_history=global_best_cost_history,
        total_evals=total_evals_so_far,
        runtime=total_runtime,
    )