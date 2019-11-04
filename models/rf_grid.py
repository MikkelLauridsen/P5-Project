from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from models.model_utility import get_standard_feature_split, find_best_hyperparameters, split_feature_label
import os


def __get_param_grid(combinations):
    parameter_grid = []
    for combination in combinations:
        parameter_grid.append(
            {'max_depth': [combination[1]],
             'n_estimators': [combination[0]],
             'criterion': ['gini', 'entropy'],
             'bootstrap': [True]}
        )

    return parameter_grid


def __get_linear_combinations(n_start, n_slope, depth_start, depth_slope, iterations=-1):
    if iterations == -1:
        iterations = int(max(n_start // abs(n_slope), depth_start // abs(depth_slope)))

    combinations = []
    for i in range(iterations):
        combinations.append([int(n_slope * i + n_start), int(depth_slope * i + depth_start)])

    return combinations


if __name__ == '__main__':
    os.chdir("..")

    # Sets up the parameter grid for random forest
    combinations = __get_linear_combinations(10, 100, 12, -1)
    print(combinations)
    print(len(combinations))

    parameter_grid = __get_param_grid(combinations)
    print("Grid for Random forest has been set up, hyper parameters are being found")

    X_train, y_train = get_standard_feature_split()

    # Finds the hyper parameters

    grid_s = GridSearchCV(RandomForestClassifier(), parameter_grid, cv=5, n_jobs=-1, scoring="f1_macro", verbose=10)
    grid_s.fit(X_train, y_train)

    print(grid_s.best_params_)
    print(grid_s.best_score_)
