import os
import datareader_csv
from sklearn.neural_network import MLPClassifier
from id_based_datasets import get_mixed_datasets
from models.model_utility import find_best_hyperparameters
from models.model_utility import scale_features
from models.model_utility import split_feature_label
from sklearn.metrics import classification_report

# Going up one directory so we have access to the below specified files.

if __name__ == "__main__":
    os.chdir("..")

    res = {}

    for i in range(1, 2):
        training_points, test_points = get_mixed_datasets(period_ms=10 * i, shuffle=True, overlap_ms=10 * i)

        print(f"Generated {len(training_points)} training points and {len(test_points)} test points")

        X_train, y_train = split_feature_label(training_points)
        X_test, y_test = split_feature_label(test_points)
        X_train, X_test = scale_features(X_train, X_test)

        parameter_space = [{'solver': ["adam"],
                        'alpha': [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06],
                        'hidden_layer_sizes': [(12, 4), (14, 4), (16, 4)],
                        'random_state': [1]}]

        #parameter_space = [{'solver': ["lbfgs", "sgd", "adam"],
        #                    'alpha': [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06],
        #                    'hidden_layer_sizes': [(8, 4), (10, 4), (12, 4), (14, 4), (16, 4)],
        #                    'random_state': [1]}]

        res[i] = (y_test, find_best_hyperparameters(MLPClassifier(), parameter_space, X_train, y_train, X_test, y_test))

    for key, pair in res:
        print(f"{10 * key}ms window: \n")
        classification_report(pair[0], pair[1])
        print("\n\n")
