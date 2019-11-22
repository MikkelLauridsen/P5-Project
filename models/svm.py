from sklearn.svm import SVC
import models.model_utility
from sklearn.model_selection import GridSearchCV
import os


def svm(parameters):  # baseline parameters
    return SVC(cache_size=3000).set_params(**parameters)


def svm_hyperparameter(X_train, y_train):
    """Finding the best hyperparameters for svm based on given training data."""
    # set up the parameter grid
    param_grid = [
        {'C': [0.01, 0.1, 1], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', "poly", "sigmoid"]}]

    # Find hyperparameters
    grid_s = GridSearchCV(SVC(), param_grid, cv=5, scoring="f1_macro", verbose=10, n_jobs=-1)
    grid_s.fit(X_train, y_train)

    print(grid_s.best_estimator_)
    print(grid_s.best_score_)
    print(grid_s.best_params_)


if __name__ == "__main__":
    os.chdir("..")

    X_train, y_train = models.model_utility.get_standard_feature_split()

    svm_hyperparameter(X_train, y_train)
