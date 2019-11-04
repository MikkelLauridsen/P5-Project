import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import models.model_utility


def lr(parameters):  # baseline parameters
    return LogisticRegression().set_params(**parameters)


def lr_hyperparameter(X_train, y_train):
    # Create regularization penalty space
    penalty = ['l1', 'l2']

    # Create regularization hyperparameter space
    C = np.logspace(0, 4, 10)

    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)

    best_model = GridSearchCV(LogisticRegression(), hyperparameters, cv=5, scoring="f1_macro", verbose=0)
    best_model.fit(X_train, y_train)
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])


if __name__ == "__main__":
    os.chdir("..")
    X_train, y_train = models.model_utility.get_standard_feature_split()
    lr_hyperparameter(X_train, y_train)
