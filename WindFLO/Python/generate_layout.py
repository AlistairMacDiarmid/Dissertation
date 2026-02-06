"""
generate valid starting positions for wind turbines in a wind farm.
uses simple rejection sampling approach to place turbines one at a time,
whilst making sure they adhear to the scenarios bounds.

starts by picking a random spot somewhere in the farm,
it then checks if the spot is within an obstacle - if it is, try again,
it then checks if the spot is too close to any turbine which has already been placed - if it is, try again,
if both these checks pass, the turbine is placed within this spot,
this is repeated until all turbines are placed.
"""

from __future__ import annotations
import numpy as np


from wind_scenario import WindScenario

def point_in_any_obstacle(
        x: float,
        y: float,
        obstacles: np.ndarray
) -> bool:
    """
    return True if point (x,y) is strictly inside any obstacle rectangle
    the obstacle shape comes in (n_obs, 4) rows [xmin,ymin,xmax,ymax]

    checks for all obstacles at once using vectorised numpy for speed.

    :param x: x-coordinate of the point to check
    :param y: y-coordinate of the point to check
    :param obstacles: array of shape (n_obs,4) with obstacle boundaries
    :return: True if the point lies inside any obstacle, False otherwise
    """

    #no obstacles check
    if obstacles.size == 0:
        return False

    #check if x and y fall in-between the edges of each obstacle
    inside_x = (x > obstacles[:, 0]) & (x < obstacles[:, 2])
    inside_y = (y > obstacles[:, 1]) & (y < obstacles[:, 3])

    #the point is inside an obstacle if both x and y are inside any obstacle
    return bool(np.any(inside_x & inside_y))


def generate_initial_feasible_layout(
        scenario: WindScenario,
        max_attempts: int = 1000000,
        seed: int | None = None
)-> np.ndarray:
    """
    generate a valid starting layout by placing turbines one at a time.

    keeps trying random positions until it finds one that confides to the farms bounds,
    the farms bounds are if it is not in any obstacles and are far away enough from already placed turbines


    :param seed:
    :param scenario: scenario with all the farms info (size, obstacles, number of turbines, etc.)
    :param max_attempts: the max attempts to try place turbines, gives up after this many attempts (avoids inifite loops)
    :return: array of shape (nturbines, 2) with all turbine (x,y) positions
    """

    #random number generator for turbine placement
    random_generator = np.random.default_rng(seed)
    # print(seed)


    #get scenario parameters
    number_of_turbines = int(scenario.nturbines)
    farm_width = float(scenario.width)
    farm_height = float(scenario.height)
    minimum_spacing_squared = float(scenario.minDist)
    obstacles = scenario.obstacles

    #empty array to store turbine placements as they are placed
    turbine_layout = np.empty((number_of_turbines, 2), dtype=float)
    turbines_placed = 0
    placement_attempts = 0

    #keep going until all turbines have been placed
    while turbines_placed < number_of_turbines:
        placement_attempts +=1
        #safety check - if turbine placement attempts have exceeded the max number of attempts
        if placement_attempts > max_attempts:
            raise RuntimeError(
                f"max number of attempted reached. failed to generate feasible layout after {max_attempts} attempts"
                f"placed {turbines_placed} turbines out of {number_of_turbines}"
            )

        #pick a random spot in the farm
        x_coordinate = float(random_generator.uniform(0, farm_width))
        y_coordinate = float(random_generator.uniform(0, farm_height))

        #check if the spot is inside an obstacle
        if point_in_any_obstacle(x_coordinate, y_coordinate, obstacles):
            continue

        #check if the spot is too close to any previously placed turbines
        if turbines_placed > 0:
            dx = turbine_layout[:turbines_placed,0] - x_coordinate
            dy = turbine_layout[:turbines_placed,1] - y_coordinate
            distance_squared = dx * dx + dy * dy
            #if any turbine is too close, reject the spot and try again
            if np.any(distance_squared < minimum_spacing_squared):
                continue

        #all checks passed, placed turbine in spot
        turbine_layout[turbines_placed] = (x_coordinate, y_coordinate)
        turbines_placed += 1

    return turbine_layout




























