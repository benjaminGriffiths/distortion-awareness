import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
import platform

import svgutils as sg
import matplotlib.lines as mlines
import matplotlib as mpl

# %% Define Key Functions / Classes
def scale_vector(x):
    y = (x - min(x)) / (max(x) - min(x))
    return y

# %% Prepare Workspace
# define datasets
n_datasets = 6

# define root directory
root_dir = ''

# load data
group_data = pd.read_csv(root_dir + 'group_data.csv')

# set style
font = {'font': 'Arial', 'size_axlabel': 6, 'size_axtick': 6, 'size_boxlabel': 5, 'size_legend': 4}
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1

# %% Plot Figures 2-4
# define iterative changes
ds_colour = [[[0.54, 0.76, 1], [0.95, 0.49, 0.50], [0.54, 0.76, 1], [0.95, 0.49, 0.50]],
             [[0.54, 0.76, 1], [0.95, 0.49, 0.50], [0.54, 0.76, 1], [0.95, 0.49, 0.50]],
             [[0.54, 0.76, 1], [0.95, 0.49, 0.50], [0.54, 0.76, 1], [0.95, 0.49, 0.50]],
             [[0.97, 0.74, 0.42], [0.94, 0.53, 0.52], [0.97, 0.74, 0.42], [0.94, 0.53, 0.52]],
             [[0.97, 0.74, 0.42], [0.94, 0.53, 0.52], [0.97, 0.74, 0.42], [0.94, 0.53, 0.52]],
             [[0.52, 0.87, 0.63], [1, 0.83, 0.54], [0.52, 0.87, 0.63], [1, 0.83, 0.54]]]
letters = [['A', 'B'], ['C', 'D'], ['E', 'F'], ['G', 'H'], ['I', 'J']]
letter_counter = 0
fig_filenames = []
fig_counter = 2
exp_label = ['Exp. 1: All Trials', 'Exp. 2: Reversed Bar', 'Exp. 2: Matched Bar', 'Exp. 3: All Trials',
             'Exp. 4: Confidence First', 'Exp. 4: Colour First', 'Exp. 5: Broad Kernel', 'Exp. 5: Narrow Kernel',
             'Exp. 6: All Trials']
exp_pos = [-88, -98, -96, -88, -103, -95, -97, -98, -88]
exp_counter = 0

# cycle through datasets
for dataset in range(n_datasets):

    # reinstaniate filenames for each figure
    if (dataset == 0) | (dataset == 3) | (dataset == 5):
        fig_filenames = []
        letter_counter = 0
        if dataset == 3:
            letter_counter += 1  # make room for hypothesis panel

    # check manipulations
    manips = np.unique(group_data.query('(dataset == {})'.format(dataset))['manipulation'])
    if (dataset == 1) | (dataset == 2):
        manips = np.flip(manips)
    n_manipulations = len(manips)

    # cycle through manipulations
    for m, manip in enumerate(manips):

        # create list for figures
        exp_filenames = [[], [], []]

        # cycle through confidence
        for confidence in [0, 1]:

            # note to self - seaborn has issues when the x- and y- axes have different limits, so this code tricks
            # seaborn into thinking the values on y-axis are twice as big as they should be. This is corrected by
            # re-labelling the y-axis tick labels.

            # get data
            pal = sns.color_palette([[0.5, 0.5, 0.5], ds_colour[dataset][confidence]])
            query = '(dataset == {}) & (confidence == {}) & (manipulation == "{}")'.format(dataset, confidence, manip)
            plot_data = group_data.query(query).drop(columns=['confidence', 'manipulation']).groupby(['pp', 'epoch'], as_index=False).mean()
            plot_data['prototype_distance'] = plot_data['prototype_distance'] * 2  # the hack!
            g = sns.jointplot(data=plot_data, x='target_distance', y='prototype_distance', hue='epoch',
                              palette=pal, ratio=5, space=0.5, marginal_ticks=False, height=3.6/2.54,
                              joint_kws={'s': 4, 'linewidths': 0.2, 'alpha': 0.3, 'edgecolor':None})
            h1 = g.ax_joint.plot([0, plot_data.target_distance[plot_data.epoch == "perception"].mean()],
                            [0, plot_data.prototype_distance[plot_data.epoch == "perception"].mean()], color=pal[0])
            h2 = g.ax_joint.plot([0, plot_data.target_distance[plot_data.epoch == "retrieval"].mean()],
                            [0, plot_data.prototype_distance[plot_data.epoch == "retrieval"].mean()], color=pal[1])
            g.ax_joint.set(xlim=[0, 0.4], ylim=[0, 0.4], xticks=[0, 0.2, 0.4], yticks=[0, 0.2, 0.4], aspect='equal')
            g.ax_joint.set_yticklabels(['0', '0.1', '0.2'], font=font['font'], fontsize=font['size_axtick']) # the fix!
            g.ax_joint.set_xticklabels(['0', '0.2', '0.4'], font=font['font'], fontsize=font['size_axlabel'])
            g.ax_joint.set_xlabel('Distance to Target', font=font['font'], fontsize=font['size_axlabel'], labelpad=1.5)
            if confidence == 1:
                g.ax_joint.set_ylabel('Distance to Prototype', font=font['font'], fontsize=font['size_axlabel'], labelpad=1.5)
            else:
                g.ax_joint.set_ylabel('', font=font['font'], fontsize=font['size_axlabel'], labelpad=1.5)
            handles, labels = g.ax_joint.get_legend_handles_labels()
            grey_dot = mlines.Line2D([], [], color=[0.5, 0.5, 0.5], marker='o', linestyle='None', markersize=1.2, label='Perception')
            col_dot = mlines.Line2D([], [], color=ds_colour[dataset][confidence], marker='o', linestyle='None', markersize=1.2, label='Retrieval')
            g.ax_joint.legend(handles=[grey_dot, col_dot], frameon=False, handletextpad=0,
                              prop=font_manager.FontProperties(family=font['font'], size=font['size_legend']))
            g.ax_joint.tick_params(axis='x', length=2, pad=1.7)
            g.ax_joint.tick_params(axis='y', length=2, pad=1.7)
            g.ax_marg_x.tick_params(axis='x', length=2, pad=1.7)
            g.ax_marg_y.tick_params(axis='y', length=2, pad=1.7)
            g.ax_marg_x.sharex(g.ax_joint)
            g.ax_marg_y.sharey(g.ax_joint)
            g.figure.subplots_adjust(left=0.192, bottom=0.192, top=0.98, right=0.98)
            fig = g.figure
            exp_filenames[confidence] = '{}/figures/fig_dataset-{}_manip-{}_conf-{}.svg'.format(root_dir, dataset, manip, confidence)
            fig.savefig(exp_filenames[confidence])
            plt.close()

        # plot interaction
        xpos = [1, 1.3, 2, 2.3]
        fig, ax = plt.subplots(1, 1, figsize=(3.8/2.54, 4/2.54))
        query = '(dataset == {}) & (manipulation == "{}")'.format(dataset, manip)
        plot_data = group_data.query(query).drop(columns='manipulation').groupby(['pp', 'epoch', 'confidence'], as_index=False).mean()
        for pp in np.unique(plot_data['pp']):
            if sum(plot_data['pp'] == pp) < 4:
                plot_data = plot_data[plot_data.pp != pp]
        dat = plot_data.pivot(index='pp', columns=['epoch', 'confidence'], values='prototypicality').values
        dat = dat[np.any(np.isnan(dat), axis=1) == 0, :]
        dat = (dat - np.tile(dat.mean(axis=1, keepdims=True), [1, 4])) + dat.mean()  # remove between subject variance (https://link.springer.com/article/10.3758/bf03210951)
        dat = dat[:, [1, 0, 3, 2]]
        g = ax.boxplot(dat, whis=np.inf, positions=xpos, patch_artist=True)
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            for gx in range(len(g[element])):
                plt.setp(g[element][gx], color=[0.5, 0.5, 0.5], linewidth=0.8)
        for patch, color in zip(g['boxes'], ds_colour[dataset]):
            patch.set_facecolor(color)
        for i, x in enumerate(xpos):
            shift = np.sign(np.mod(i, 2)-0.9)
            kernel = scipy.stats.gaussian_kde(dat[:, i]).pdf(np.linspace(0, 1, 100))
            kernel = scale_vector(kernel)
            scatter_pos = np.zeros(len(dat))
            for j in range(len(dat[:, i])):
                scatter_pos[j] = x + (((np.random.rand(1) * kernel[np.round(dat[j, i]*100).astype('int')]) * 0.08 + 0.16) * shift)
            ax.scatter(scatter_pos, dat[:, i], zorder=10, color=ds_colour[dataset][i], s=0.5, linewidths=0.5, alpha=0.5)
        ax.set(ylim=[0, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xticks=[np.mean(xpos[:2]), np.mean(xpos[2:])],
               xticklabels=['perception', 'retrieval'])
        ax.set_ylabel(ylabel='Prototypicality (arb. units)', font=font['font'], fontsize=font['size_axlabel'], labelpad=3)
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], font=font['font'], fontsize=font['size_axtick'])
        ax.set_xticklabels(['Perception', 'Retrieval'], font=font['font'], fontsize=font['size_boxlabel'])
        ax.tick_params(axis='y', length=2, pad=1.7)
        ax.tick_params(axis='x', length=0, pad=2)
        ax.spines['bottom'].set_alpha(0)
        sns.despine(fig)
        grey_dot = mlines.Line2D([], [], color=ds_colour[dataset][0], marker='o', linestyle='None', markersize=1.2, label='Sure')
        col_dot = mlines.Line2D([], [], color=ds_colour[dataset][1], marker='o', linestyle='None', markersize=1.2, label='Not Sure')
        ax.legend(handles=[grey_dot, col_dot], frameon=False, handletextpad=0, loc='lower right',
                  prop=font_manager.FontProperties(family=font['font'], size=font['size_legend']))
        plt.tight_layout()
        plt.show()

        exp_filenames[2] = '{}/figures/fig_dataset-{}_manip-{}_interaction.svg'.format(root_dir, dataset, manip)
        fig.savefig(exp_filenames[2])
        plt.close()

        # report labels (for debugging)
        print('plotting "{}" using {}'.format(exp_label[exp_counter], manip))

        # combine figures
        fig_filenames.append('{}/figures/fig_dataset-{}_manip-{}.svg'.format(root_dir, dataset, manip))
        curr_letters = letters[letter_counter]
        sg.compose.Figure(380, 120,
               sg.compose.SVG(exp_filenames[1]).scale(1.2).move(15, 0),
               sg.compose.Text(exp_label[exp_counter], exp_pos[exp_counter], 11, font='Arial', size=8, weight='bold').rotate(270),
               sg.compose.Text(curr_letters[0], 0, 8, font='Arial', size=10, weight='bold'),
               sg.compose.Panel(sg.compose.SVG(exp_filenames[0]).scale(1.2)).move(135, 0),
               sg.compose.Panel(sg.compose.SVG(exp_filenames[2]).scale(1.2)).move(260, -6),
               sg.compose.Text(curr_letters[1], 261, 8, font='Arial', size=10, weight='bold'),
               ).save(fig_filenames[-1])
        exp_counter += 1
        letter_counter += 1

    # create figures
    if dataset == 2:
        sg.compose.Figure(380, 140*len(fig_filenames)-20,
                          sg.compose.SVG(fig_filenames[0]).move(0, 5),
                          sg.compose.Text('"Sure"', 65, 6, font='Arial', size=8, weight='bold'),
                          sg.compose.Text('"Not Sure"', 178, 6, font='Arial', size=8, weight='bold'),
                          sg.compose.Line([[0, 0], [370, 0]], width=1, color='#c0c0c0').move(0, 132),
                          sg.compose.SVG(fig_filenames[1]).move(0, 140),
                          sg.compose.SVG(fig_filenames[2]).move(0, 270),
                          sg.compose.Line([[0, 0], [370, 0]], width=1, color='#c0c0c0').move(0, 395),
                          sg.compose.SVG(fig_filenames[3]).move(0, 405),
                          ).save('{}/figures/fig2.svg'.format(root_dir, fig_counter))

    elif dataset == 4:
        sg.compose.Figure(370, 140*len(fig_filenames)-20,
                          sg.compose.SVG(fig_filenames[0]).move(0, 5),
                          sg.compose.Text('"Sure"', 65, 6, font='Arial', size=8, weight='bold'),
                          sg.compose.Text('"Not Sure"', 178, 6, font='Arial', size=8, weight='bold'),
                          sg.compose.SVG(fig_filenames[1]).move(0, 135),
                          sg.compose.Line([[0, 0], [370, 0]], width=1, color='#c0c0c0').move(0, 263),
                          sg.compose.SVG(fig_filenames[2]).move(0, 275),
                          sg.compose.SVG(fig_filenames[3]).move(0, 405),
                          ).save('{}/figures/fig3.svg'.format(root_dir, fig_counter))

    elif dataset == 5:
        sg.compose.Figure(370, 130*len(fig_filenames),
                          sg.compose.SVG(fig_filenames[0]).move(0, 5),
                          sg.compose.Text('"Sure"', 65, 6, font='Arial', size=8, weight='bold'),
                          sg.compose.Text('"Not Sure"', 178, 6, font='Arial', size=8, weight='bold')
                          ).save('{}/figures/fig4.svg'.format(root_dir, fig_counter))