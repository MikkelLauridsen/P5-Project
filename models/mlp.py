from sklearn.neural_network import MLPClassifier
import os
import models.model_utility
from sklearn.model_selection import GridSearchCV


def mlp(parameters):  # baseline parameters
    return MLPClassifier(hidden_layer_sizes=(12, 2), max_iter=500).set_params(**parameters)


def find_mlp_parameters():
    os.chdir("..")

    X, y = models.model_utility.get_standard_feature_split()

    parameter_space = [
        {'activation': ['logistic', 'relu'],
         'solver': ['lbfgs', 'sgd', 'adam'],
         'alpha': [1.e-03, 1.e-04, 1.e-05, 1.e-06],
         'learning_rate': ['constant', 'adaptive'],
         'hidden_layer_sizes': [(12, 3), (14, 3), (16, 3), (12, 4), (14, 4), (16, 4)],
         'max_iter': [200, 400, 600]}]

    classifier = GridSearchCV(mlp(), parameter_space, cv=5, n_jobs=-1, scoring="f1_macro", verbose=10).fit(X, y)
    best_parameters = classifier.best_params_

    print("MLP parameters found:")

    for parameter in best_parameters:
        print(f"\t{parameter} = {best_parameters[parameter]}")

    with open("mlp.txt", "w", newline="") as file:
        for parameter in best_parameters:
            file.write(f"{parameter} = {best_parameters[parameter]}\n")

    return best_parameters


if __name__ == "__main__":
    find_mlp_parameters()
