"""Functions for generating results by running the models with different settings."""
import os
import time
import concurrent.futures as conf
import models.model_utility as utility
import hugin.pyhugin87 as hugin
from models.model_utility import get_scaled_training_validation
from metrics import get_metrics, get_metrics_path, get_error_metrics
from datapoint import datapoint_attributes
from datareader_csv import load_metrics
from datawriter_csv import save_metrics, save_time


def generate_validation_results(windows=None, strides=None, imp_splits=None,
                                dos_types=None, models=None, eliminations=0):
    """
    Generates and saves metrics for all combinations of specified parameters using the validation set.
    If metrics for a given combination already exist, this combination is skipped.

    :param windows: A list of feature window sizes (int ms)
    :param strides: A list of stride sizes (int ms)
    :param imp_splits: A list of impersonation type options (True, False)
    :param dos_types: A list of DoS type options ('modified', 'original')
    :param models: A dictionary of model names to parameter spaces ({'**model**': {'**parameter**': [**values**]}}
    :param eliminations: The number of features eliminated in step-wise elimination
    :return:
    """

    windows = [100] if windows is None else windows
    strides = [100] if strides is None else strides
    imp_splits = [True] if imp_splits is None else imp_splits
    dos_types = ['modified'] if dos_types is None else dos_types
    models = {'mlp': {}} if models is None else models

    # Calculate number of jobs to be conducted.
    inner_loop_size = __get_stepwise_size(eliminations) * len(models) if eliminations > 0 else len(models)
    job_count = len(windows) * len(strides) * len(imp_splits) * len(dos_types) * inner_loop_size
    current_job = 0

    for period_ms in windows:
        for stride_ms in strides:
            for imp_split in imp_splits:
                for dos_type in dos_types:
                    # Get datasets for current combination of parameters.
                    X_train, y_train, X_validation, y_validation, feature_time_dict = get_scaled_training_validation(
                        period_ms, stride_ms,
                        imp_split, dos_type)

                    print(f"starting jobs {current_job} through {current_job + inner_loop_size} of "
                          f"{job_count} -- {(current_job / job_count) * 100.0}%")

                    if eliminations > 0:
                        # Conduct stepwise-elimination to test different feature subsets.
                        __save_stepwise_elimination(
                            models,
                            X_train, y_train,
                            X_validation, y_validation,
                            eliminations,
                            feature_time_dict,
                            period_ms, stride_ms,
                            imp_split, dos_type)
                    else:
                        # Use the full feature poll.
                        subset = list(datapoint_attributes)[2:]

                        for model in models.keys():
                            create_and_save_results(
                                model, models[model],
                                X_train, y_train,
                                X_validation, y_validation,
                                feature_time_dict,
                                period_ms, stride_ms,
                                imp_split, dos_type,
                                subset)

                    current_job += inner_loop_size


def __get_stepwise_size(max_features):
    # Returns the number of feature subsets to be tested.

    n = len(list(datapoint_attributes)[2:])
    size = 0

    for i in range(max_features):
        size += n
        n -= 1

    return size


def create_and_save_results(model, parameters, X_train, y_train, X_test, y_test, feature_time_dict,
                            period_ms, stride_ms, imp_split, dos_type, subset, is_test=False):
    """
    Runs specified model with specified parameters on specified dataset and saves the result to file.

    :param model: Model name ('bn', 'dt', 'knn', 'lr', 'mlp', 'nbc', 'rf', 'svm')
    :param parameters: Model parameter space {'**parameter**': [**values**]}
    :param X_train: Training set feature values
    :param y_train: Training set class labels
    :param X_test: Test or validation set feature values
    :param y_test: Test or validation set class labels
    :param feature_time_dict: A dictionary of {'**feature**': **time_ns**}
    :param period_ms: Window size (int ms)
    :param stride_ms: Stride size (int ms)
    :param imp_split: The impersonation type (True, False)
    :param dos_type: The DoS type ('modified', 'original')
    :param subset: A list of labels of features to be used
    :param is_test: A flag indicating whether the test set or validation set is used
    :return:
    """
    path, _ = get_metrics_path(
        period_ms, stride_ms,
        imp_split, dos_type,
        model, parameters == {},
        subset, is_test=is_test)

    metric_type = "test" if is_test else "validation"

    if os.path.exists(path):
        metrics = load_metrics(
            period_ms, stride_ms,
            imp_split, dos_type,
            model, parameters == {},
            subset, is_test=is_test)

        print(f"Skipped existing {metric_type} metrics at {path}")
    else:
        try:
            X_train_mod = __create_feature_subset(X_train, subset)
            X_test_mod = __create_feature_subset(X_test, subset)

            classifier = utility.get_classifier(model, parameters, subset)
            classifier.fit(X_train_mod, y_train)

            before = time.perf_counter_ns()
            y_predict = classifier.predict(X_test_mod)
            time_model = (time.perf_counter_ns() - before) / len(X_test_mod)

            # Calculate scores on test set.
            metrics = get_metrics(y_test, y_predict)

            save_metrics(metrics, period_ms, stride_ms, imp_split, dos_type, model, parameters, subset, is_test=is_test)
            time_feature = 0.0

            # Find sum of feature times.
            for feature in feature_time_dict.keys():
                if feature in subset:
                    time_feature += feature_time_dict[feature]

            save_time(
                time_model, time_feature,
                period_ms, stride_ms,
                imp_split, dos_type,
                model, parameters,
                subset, is_test=is_test)

            print(f"Saved {metric_type} metrics to {path}")

        except hugin.HuginException:
            print(f"Failed generating results for {model} at {path}")

            metrics = get_error_metrics()

    return metrics


def __save_stepwise_elimination(models, X_train, y_train, X_validation, y_validation, max_features,
                                feature_time_dict, period_ms, stride_ms, imp_split, dos_type):
    # Runs step-wise elimination on specified parameters and saves the results of each subset model combination.

    # Get feature labels.
    labels = (list(datapoint_attributes)[2:]).copy()
    working_set = labels.copy()

    for i in range(max_features):
        # Save the feature label which yields the best result when eliminated from the pool.
        best_score = 0
        best_label = ""

        for label in labels:
            current_subset = working_set.copy()
            del current_subset[current_subset.index(label)]

            with conf.ProcessPoolExecutor() as executor:
                futures = {executor.submit(
                    create_and_save_results,
                    model, models[model],
                    X_train, y_train,
                    X_validation, y_validation,
                    feature_time_dict,
                    period_ms, stride_ms,
                    imp_split, dos_type,
                    current_subset) for model in models.keys()}

                for future in conf.as_completed(futures):
                    score = future.result()['total'].f1

                    if score > best_score:
                        best_score = score
                        best_label = label

        # Remove the feature label which yields the best result when eliminated from the pool.
        del working_set[working_set.index(best_label)]
        del labels[labels.index(best_label)]


def __create_feature_subset(X, subset):
    # Returns a modified copy of input list of feature values,
    # which only contains values of features in the specified subset.

    indices = [list(datapoint_attributes)[2:].index(f) for f in subset]
    length = len(list(datapoint_attributes)[2:])

    X_mod = []

    for sample in X:
        sample_mod = []

        for i in range(0, length):
            if i in indices:
                sample_mod.append(sample[i])

        X_mod.append(sample_mod)

    return X_mod


selected_models = {
    'bn': {},
    'nbc': {},
    'mlp': {
        'activation': 'logistic',
        'alpha': 0.0001,
        'hidden_layer_sizes': (16, 3),
        'learning_rate': 'adaptive',
        'max_iter': 600,
        'solver': 'lbfgs'},

    'svm': {
        'C': 1000,
        'gamma': 0.1,
        'kernel': 'rbf'},

    'knn': {
        'metric': 'manhattan',
        'n_neighbors': 8,
        'weights': 'distance'},

    'lr': {
        'C': 3593.813663804626,
        'penalty': 'l2'},

    'dt': {
        'criterion': 'entropy',
        'max_depth': 13,
        'min_samples_split': 3},

    'rf': {
        'bootstrap': True,
        'criterion': 'gini',
        'max_depth': 11,
        'n_estimators': 110}
}

if __name__ == "__main__":


    generate_validation_results(
        windows=[100, 50, 20, 10],
        strides=[100, 50, 20, 10],
        imp_splits=[False],
        dos_types=['modified'],
        models=selected_models,
        eliminations=4)
