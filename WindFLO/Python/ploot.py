import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#load full experiment history
df = pd.read_csv('Results_obs_05/history/experiment_history.csv')
#degfine the scenario
df = df[df['scenario'] == 'obs_05.xml'].copy()
#scale cost values
df['best_cost_so_far'] = df['best_cost_so_far'] * 1e6  # scale to x10^-6

#fixed order for consistency
ALGO_ORDER = ['ga', 'hc', 'shc', 'rrhc']
#labels for subplot titles
ALGO_LABELS = {'ga': 'GA', 'hc': 'HC', 'shc': 'SHC', 'rrhc': 'RRHC'}

#colour mapping
COLOR = {'ga': '#2c6fad', 'hc': '#e07b39', 'shc': '#4caf72', 'rrhc': '#c0392b'}


#create one subplot per algorithm, all with same y-axis
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
#main title
fig.suptitle('Convergence per Run by Algorithm (obs_05.xml)', fontsize=14, fontweight='bold', y=1.01)

#loop through each algorithm and draw thier runs on separate subplots
for ax, algo in zip(axes, ALGO_ORDER):
    algo_df = df[df['algorithm'] == algo]
    #get all random seeds used
    seeds = algo_df['seed'].unique()

    #compute median run by final cost
    final_costs = algo_df.groupby('seed')['best_cost_so_far'].last()
    #select the seeds whose final cost sits at the median
    median_seed = final_costs.index[(final_costs.rank(method='first') - 1 == len(seeds) // 2).values]

    #if no median is found, highlight best run
    if len(median_seed) == 0:
        median_seed = [final_costs.idxmin()]

    #plot every run for current algorithm
    for seed in seeds:
        run = algo_df[algo_df['seed'] == seed]
        #check if therun is the chosen median
        is_median = seed in median_seed
        #draw median run with stronger line, others are faintly drawn
        ax.plot(
            run['evaluation'],
            run['best_cost_so_far'],
            color=COLOR[algo],
            alpha=0.85 if is_median else 0.2,
            linewidth=2.0 if is_median else 0.9,
            zorder=3 if is_median else 2,
            label='Median run' if is_median else '_nolegend_'
        )

    #subplot title
    ax.set_title(ALGO_LABELS[algo], fontsize=13, fontweight='bold')
    #x-axis label for each subplot
    ax.set_xlabel('Objective-function evaluations', fontsize=10)
    #tick labels slightly smaller
    ax.tick_params(axis='both', labelsize=9)
    #enable the grid
    ax.grid(True, linestyle='--', alpha=0.4)
    #format y-axis to 1 decimal place
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    #create the median run legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR[algo], linewidth=2.0, label='Median run'),
        Line2D([0], [0], color=COLOR[algo], linewidth=0.9, alpha=0.4, label='Individual runs'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

#only set the first subplot to contain the y-axis label
axes[0].set_ylabel('Best-so-far energy cost (× 10⁻⁶)', fontsize=10)

#adjust spacing
plt.tight_layout()
#save final plot to file
plt.savefig('convergence_faceted.png', dpi=180, bbox_inches='tight')
print("saved.")