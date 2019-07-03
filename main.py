#!/usr/bin/env python3

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

from tools import prepare_signal_background, prepare_wXy, get_data_X, get_metrics
from plotting import plot_sig_vs_bg, plot_decision_function, plot_data_decision_function
from optimiser import Optimiser


data_file = 'data/data.root'

signal_trees = ['signal_tree_training;1', 'signal_tree_testing;1']
background_trees = ['background_tree_training;1', 'background_tree_testing;1']

data_tree = 'Data;1'

variable_names = ['mll', 'dRll', 'pTll']

plot_dir = 'plots'


def make_plots():
    signal, background = prepare_signal_background(
        data_file, signal_trees, background_trees, variable_names)

    # === make signal vs. bg plots ===

    plot_sig_vs_bg(signal, background, plot_dir)

    # === 5 fold cross-validation ===

    w, X, y = prepare_wXy(signal, background, variable_names)

    bdt = GradientBoostingClassifier(learning_rate=0.005,
                                     max_depth=5,
                                     n_estimators=100)

    kf = KFold(n_splits=2)

    bdt_output = []
    classes = []
    weights = []

    for train, test in kf.split(X):
        X_train, y_train, w_train = X[train], y[train], w[train]
        X_test, y_test, w_test = X[test], y[test], w[test]

        bdt.fit(X_train, y_train, w_train)

        bdt_output.append(
            bdt.decision_function(X_test)
        )

        classes.append(y_test)
        weights.append(w_test)

    bdt_output = np.concatenate(bdt_output)
    classes = np.concatenate(classes)
    weights = np.concatenate(weights)

    # === plot decision functions ===

    plot_decision_function(decision_function=bdt_output,
                           classes=classes,
                           weights=weights,
                           plotting_dir=plot_dir)

    data_X = get_data_X(data_file, data_tree, variable_names)

    bdt.fit(X, y, w)
    data_decision_function = bdt.decision_function(data_X)

    plot_data_decision_function(data_decision_function, plot_dir)

    # === estimate the signal fraction ===

    tpr, tnr = get_metrics(bdt_output, classes, weights)

    n_data = len(data_decision_function)
    n_minus = sum(data_decision_function < 0)

    x = (n_minus / n_data - tnr) / (1 - tnr - tpr)

    print(
        ('\n=========== Results ===========\n'
         ' Signal event fraction\n'
         ' x: %.3f' % x + '\n'
         '===============================\n')
    )


def optimise():
    signal, background = prepare_signal_background(
        data_file, signal_trees, background_trees, variable_names)

    w, X, y = prepare_wXy(signal, background, variable_names)

    hyperparameter_settings = {
        'learning_rate': [0.001, 0.01]
    }

    optimiser = Optimiser(
        GradientBoostingClassifier, X, y,
        hyperparameter_settings=hyperparameter_settings,
        fixed_hyperparameters={'max_depth': 8, 'loss': 'deviance'},
        n_iterations=20
    )

    optimiser.run()


if __name__ == '__main__':
    # optimise()
    make_plots()
