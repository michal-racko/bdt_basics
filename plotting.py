import os
import numpy as np

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches


def plot_sig_vs_bg(signal: dict,
                   background: dict,
                   plotting_dir: str):
    current_dir = ''

    for d in plotting_dir.split('/'):
        current_dir += d + '/'

        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)

    patches = [mpatches.Patch(color='tab:blue', label='background'),
               mpatches.Patch(color='tab:orange', label='signal')]

    for i, v in enumerate(signal):
        if v == 'wei':
            continue

        if v not in background:
            raise Exception('Variable ' + v + ' not found in background')

        pl.figure(i)

        pl.hist(
            [signal[v], background[v]],
            weights=[signal['wei'], background['wei']],
            color=['tab:orange', 'tab:blue'],
            density=True,
            rwidth=0.9,
            bins=20
        )

        pl.legend(handles=patches, fontsize=16, loc=1)

        pl.title(v, fontsize=20)
        pl.savefig(plotting_dir + '/' + v + '.png')
        pl.close()


def plot_decision_function(decision_function: np.ndarray,
                           classes: np.ndarray,
                           weights: np.ndarray,
                           plotting_dir: str):
    current_dir = ''

    for d in plotting_dir.split('/'):
        current_dir += d + '/'

        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)

    signal_decision_function = decision_function[classes == 1]
    signal_weights = weights[classes == 1]

    background_decision_function = decision_function[classes == 0]
    background_weights = weights[classes == 0]

    patches = [mpatches.Patch(color='tab:blue', label='signal'),
               mpatches.Patch(color='tab:red', label='background')]

    pl.figure()

    pl.hist(
        [signal_decision_function, background_decision_function],
        weights=[signal_weights, background_weights],
        color=['tab:blue', 'tab:red'],
        density=True,
        rwidth=0.9,
        bins=20
    )

    pl.legend(handles=patches, fontsize=16, loc=1)

    pl.title('BDT decision function', fontsize=20)
    pl.savefig(plotting_dir + '/decision_function.png')
    pl.close()


def plot_data_decision_function(data_decision_function: np.ndarray,
                                plotting_dir: str):
    current_dir = ''

    for d in plotting_dir.split('/'):
        current_dir += d + '/'

        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)

    patches = [mpatches.Patch(color='tab:green', label='data')]

    pl.figure()

    pl.hist(
        data_decision_function,
        color='tab:green',
        rwidth=0.9,
        bins=20
    )

    pl.legend(handles=patches, fontsize=16, loc=1)

    pl.title('BDT decision function for data', fontsize=20)
    pl.savefig(plotting_dir + '/decision_function_data.png')
    pl.close()