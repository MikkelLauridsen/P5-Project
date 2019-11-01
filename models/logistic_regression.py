import os
import numpy as np
from sklearn.linear_model import LogisticRegression, logistic
import models.model_utility
from sklearn.model_selection import GridSearchCV

def lr():  # baseline parameters
    return LogisticRegression()


def lgr(X_train, y_train):
    # Create regularization penalty space
    penalty = ['l1', 'l2']

    # Create regularization hyperparameter space
    C = np.logspace(0, 4, 10)

    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)

    best_model = GridSearchCV(logistic(), hyperparameters, cv=5, scoring="f1_macro")
    best_model.fit(X_train, y_train)
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])
    print(best_model.best_params_)


if __name__ == "__main__":
    os.chdir("..")
    X_train, y_train = models.model_utility.get_standard_feature_split()
    lgr(X_train, y_train)

#def logistic_regression(X_train, y_train):
#    # Define the model
#    print("the Logistic Regression model is now being created and fitted with data")
#    logistic_regr = LogisticRegression()
#
#    # Fit the model
#    logistic_regr.fit(X_train, y_train)
#
#    # Model get returned so accuracy can get calculated
#    print("The Logistic Regression Model has been created, prediction of accuracy is now being calculated")
#    return logistic_regr
