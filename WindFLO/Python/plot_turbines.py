import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import numpy as np

"""
add title parameter 

"""


def plot_turbines(scenario, best_layout, title = None):

    x = best_layout[:,0]
    y = best_layout[:,1]


    plt.figure(figsize=(10,10))
    plt.scatter(x,y,s=10,alpha=0.5, label='Turbines')


    plt.xlabel("farm width (m)")
    plt.ylabel("farm height (m)")
    plt.title(title)

    plt.xlim(0, scenario.width)
    plt.ylim(0, scenario.height)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)


    if scenario.obstacles.size != 0:
        for obs in scenario.obstacles:
            xmin, ymin, xmax, ymax = obs
            plt.gca().add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=2,
                    label="obstacles"
                )
            )


    plt.show()



