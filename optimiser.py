import json
import numpy as np

from math import ceil
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier


class Optimiser(object):
    """
    Can be used to optimise hyperparameters
    of a multivariate technique
    """

    def __init__(self,
                 classfifier_class: object,
                 X: np.ndarray,
                 y: np.ndarray,
                 hyperparameter_settings: dict,
                 fixed_hyperparameters={},
                 metric_setting='separation',
                 strategy='grid',
                 weights=None,
                 n_bins=20,
                 n_splits=3,
                 n_iterations=10,
                 output_path='optimisation_results.json'):
        """
        hyperparameter_settings should be a dict of the form:
            {
                <name>: [min_value, max_value]
            }
        """
        self.X = X
        self.y = y

        if weights is None:
            self.weights = np.ones(len(y))
        else:
            self.weights = weights

        self.classfifier_class = classfifier_class

        self.metric_setting = metric_setting

        self.strategy = strategy

        self.n_bins = n_bins
        self.n_splits = n_splits
        self.n_iterations = n_iterations

        self.i = 0

        self.hyperparameter_space = {}

        self._prepare_hyperparameter_space(hyperparameter_settings)
        self.fixed_hyperparameters = fixed_hyperparameters

        self.results = {}

        self.output_path = output_path

    def _eval_performance(self):
        print('Step', self.i)

        if self.i >= len(list(self.hyperparameter_space.values())[0]):
            return False

        current_hyperparameters = {
            k: self.hyperparameter_space[k][self.i] for k in self.hyperparameter_space
        }
        current_hyperparameters.update(self.fixed_hyperparameters)

        print('INFO: evaluating performance for', current_hyperparameters)

        classifier = self.classfifier_class(**current_hyperparameters)

        kf = KFold(n_splits=5)

        y = []
        weights = []
        decision_function = []

        for train, test in kf.split(self.X):
            X_train, y_train, w_train = self.X[train], self.y[train], self.weights[train]
            X_test, y_test, w_test = self.X[test], self.y[test], self.weights[test]

            classifier.fit(X_train, y_train, w_train)

            decision_function.append(classifier.decision_function(X_test))
            weights.append(w_test)
            y.append(y_test)

        y = np.concatenate(y)
        weights = np.concatenate(weights)
        decision_function = np.concatenate(decision_function)

        if self.metric_setting.lower() == 'separation':
            score = self._separation(y, weights, decision_function)
        else:
            raise Exception('Unknown metric setting ' + self.metric_setting)

        self.results[self.i] = {
            'hyperparameters': current_hyperparameters,
            'score': score
        }

        self._save_results()

        self.i += 1

        return True

    def _prepare_hyperparameter_space(self,
                                      hyperparameter_settings: dict):
        points_per_dimension = ceil(
            self.n_iterations ** (1 / len(hyperparameter_settings)))

        if self.strategy.lower() == 'grid':
            hyperparameters = [h for h in hyperparameter_settings]

            params = tuple(
                np.linspace(
                    hyperparameter_settings[h][0],
                    hyperparameter_settings[h][1],
                    points_per_dimension
                )
                for h in hyperparameters
            )

            hyperparam_values = np.meshgrid(*params)

            for i, hyperparameter in enumerate(hyperparameters):
                self.hyperparameter_space[hyperparameter] = hyperparam_values[i].ravel(
                )

        else:
            raise Exception('Unknown strategy \"' + self.strategy + '\"')

    def _save_results(self):
        json_string = json.dumps(self.results)

        with open(self.output_path, 'w') as f:
            f.write(json_string)

    def _separation(self,
                    y: np.ndarray,
                    weights: np.ndarray,
                    decision_function: np.ndarray):
        hist_range = [min(decision_function), max(decision_function)]

        signal_hist = np.histogram(
            decision_function[y == 1],
            bins=self.n_bins,
            weights=weights[y == 1]
        )[0]

        signal_hist /= sum(signal_hist)

        background_hist = np.histogram(
            decision_function[y == 0],
            bins=self.n_bins,
            weights=weights[y == 0]
        )[0]

        background_hist /= sum(background_hist)

        return sum((signal_hist - background_hist) ** 2 / (signal_hist + background_hist)) / 2

    def run(self):
        while True:
            if not self._eval_performance():
                return
