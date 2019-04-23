import os
import uproot

import numpy as np


def prepare_signal_background(filepath: str,
                             singal_trees: list,
                             background_trees: list,
                             variables: list):
    """
    Prepares np.ndarras for each variable for both
    signal and background. Weight is also added to the
    output dict.
    """
    uproot_file = uproot.open(filepath)

    variables += ['wei']

    for t in singal_trees + background_trees:
        if bytes(t, 'UTF-8') not in uproot_file.keys():
            raise Exception(
                'Tree \"' + t + '\" not found in the root file at ' + filepath)

    signal_variables = {}
    background_variables = {}

    for v in variables:
        signal_values = []
        background_values = []

        for t in singal_trees:
            if bytes(v, 'UTF-8') not in uproot_file[bytes(t, 'UTF-8')].keys():
                raise Exception(v + ' not found in the ' + t + ' tree')

            signal_values.append(
                uproot_file[bytes(t, 'UTF-8')][bytes(v, 'UTF-8')].array())

        for t in background_trees:
            if bytes(v, 'UTF-8') not in uproot_file[bytes(t, 'UTF-8')].keys():
                raise Exception(v + ' not found in the ' + t + ' tree')

            background_values.append(
                uproot_file[bytes(t, 'UTF-8')][bytes(v, 'UTF-8')].array())

        signal_variables[v] = np.concatenate(signal_values)
        background_variables[v] = np.concatenate(background_values)

    return signal_variables, background_variables


def prepare_wXy(signal, background, variables):
    values = []

    for i, v in enumerate(variables):
        if v not in background:
            raise Exception('Variable ' + v + ' not found in background')

        if v == 'wei':
            w = np.concatenate([
                signal[v],
                background[v]
            ])
        else:
            values.append(np.concatenate([
                signal[v],
                background[v]
            ]))

        if i == 0:
            signal_len = len(signal[v])

    X = np.vstack(values).T
    y = np.zeros(len(X))
    y[:signal_len] = 1

    return w, X, y


def get_data_X(filepath: str,
               tree: str,
               variables: list):
    uproot_file = uproot.open(filepath)

    if bytes(tree, 'UTF-8') not in uproot_file.keys():
        raise Exception(
            'Tree \"' + tree + '\" not found in the root file at ' + filepath)

    values = []

    for v in variables:
        if v == 'wei':
            continue

        if bytes(v, 'UTF-8') not in uproot_file[bytes(tree, 'UTF-8')].keys():
            raise Exception(v + ' not found in the ' + tree + ' tree')

        values.append(
            uproot_file[bytes(tree, 'UTF-8')][bytes(v, 'UTF-8')].array())

    return np.vstack(values).T


def get_metrics(decision_function: np.ndarray,
                classes: np.ndarray,
                weights: np.ndarray,
                threshold=0):
    bg = decision_function[classes == 0]
    bg_wei = weights[classes == 0]

    sig = decision_function[classes == 1]
    sig_wei = weights[classes == 1]

    tnr = sum((bg < 0) * bg_wei) / sum(bg_wei)
    tpr = sum((sig > 0) * sig_wei) / sum(sig_wei)

    return tpr, tnr
