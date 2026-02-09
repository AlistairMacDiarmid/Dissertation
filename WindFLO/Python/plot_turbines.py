import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



def plot_turbines(scenario, best_layout, title = None):

     # x = best_layout[:,0]
     # y = best_layout[:,1]

    df = pd.DataFrame({
        "x": best_layout[:,0],
        "y": best_layout[:,1],
    })

    sns.set_theme(style="whitegrid")

    ax = sns.scatterplot(
         data=df,
         x="x",
         y="y",
         label="turbines",
     )

    ax.set_title(title)
    ax.set_xlabel("farm width (m)")
    ax.set_ylabel("farm height (m)")
    plt.xlim(0, scenario.width)
    plt.ylim(0, scenario.height)

    if scenario.obstacles.size != 0:
        for obs in scenario.obstacles:
            xmin, ymin, xmax, ymax = obs
            obst_rect = plt.Rectangle(
                (xmin,ymin),
                xmax-xmin,
                ymax-ymin,
                fill=False,
                edgecolor='black',
                linewidth=1,
                label='obstacle' if scenario.obstacles.size == 0 else None,
            )
            ax.add_patch(obst_rect)

    plt.show()

    #
    #
    # plt.figure(figsize=(10,10))
    # plt.scatter(x,y,s=10,alpha=0.5, label='Turbines')
    #
    #
    # plt.xlabel("farm width (m)")
    # plt.ylabel("farm height (m)")
    # plt.title(title)
    #
    # plt.xlim(0, scenario.width)
    # plt.ylim(0, scenario.height)
    #
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)
    #
    #
    # if scenario.obstacles.size != 0:
    #     for obs in scenario.obstacles:
    #         xmin, ymin, xmax, ymax = obs
    #         plt.gca().add_patch(
    #             plt.Rectangle(
    #                 (xmin, ymin),
    #                 xmax - xmin,
    #                 ymax - ymin,
    #                 fill=False,
    #                 edgecolor="yellow",
    #                 linewidth=2,
    #                 label="obstacles"
    #             )
    #         )
    #
    #
    # plt.show()



