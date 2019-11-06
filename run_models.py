"""Functions for generating results by running the models with different settings."""
import concurrent.futures as conf
import os
import time

import models.model_utility as utility
from datapoint import datapoint_attributes
from datareader_csv import load_metrics
from datawriter_csv import save_metrics, save_time
from metrics import get_metrics, get_metrics_path


def generate_results(windows=[100], strides=[100], imp_splits=[True],
                     dos_types=['modified'], models={'mlp': {}}, eliminations=0):
    """
    Generates and saves metrics for all combinations of specified parameters. If metrics for a given combination
    already exist, this combination is skipped.

    :param windows: A list of feature window sizes (int ms)
    :param strides:   A list of stride sizes (int ms)
    :param imp_splits:   A list of impersonation type options (True, False)
    :param dos_types:    A list of DoS type options ('modified', 'original')
    :param models:       A dictionary of model names to parameter spaces ({'**model**': {'**parameter**': [**values**]}}
    :param eliminations: The number of features eliminated in step-wise elimination
    :return:
    """
    # calculate number of jobs to be conducted
    inner_loop_size = __get_stepwise_size(eliminations) * len(models) if eliminations > 0 else len(models)
    job_count = len(windows) * len(strides) * len(imp_splits) * len(dos_types) * inner_loop_size
    current_job = 0

    for period_ms in windows:
        for stride_ms in strides:
            for imp_split in imp_splits:
                for dos_type in dos_types:
                    # get datasets for current combination of parameters
                    X_train, y_train, X_test, y_test, feature_time_dict = utility.get_dataset(
                        period_ms, stride_ms,
                        imp_split, dos_type)

                    print(f"starting jobs {current_job} through {current_job + inner_loop_size} of "
                          f"{job_count} -- {current_job / job_count}%")

                    if eliminations > 0:
                        # conduct stepwise-elimination to test different feature subsets
                        __save_stepwise_elimination(
                            models,
                            X_train, y_train,
                            X_test, y_test,
                            eliminations,
                            feature_time_dict,
                            period_ms, stride_ms,
                            imp_split, dos_type)
                    else:
                        # use the full feature poll
                        subset = list(datapoint_attributes)[2:]

                        for model in models.keys():
                            create_and_save_results(
                                model, models[model],
                                X_train, y_train,
                                X_test, y_test,
                                feature_time_dict,
                                period_ms, stride_ms,
                                imp_split, dos_type,
                                subset)

                    current_job += inner_loop_size


def __get_stepwise_size(max_features):
    # returns the number of feature subsets to be tested

    n = len(list(datapoint_attributes)[2:])
    size = 0

    for i in range(max_features):
        size += n
        n -= 1

    return size


def create_and_save_results(model, parameters, X_train, y_train, X_test, y_test, feature_time_dict,
                            period_ms, stride_ms, imp_split, dos_type, subset):
    """
    Runs specified model with specified parameters on specified dataset and saves the result to file.

    :param model: Model name ('bn', 'dt', 'knn', 'lr', 'mlp', 'nbc', 'rf', 'svm')
    :param parameters: Model parameter space {'**parameter**': [**values**]}
    :param X_train: Training set feature values
    :param y_train: Training set class labels
    :param X_test: Test set feature values
    :param y_test: Test set class labels
    :param feature_time_dict: A dictionary of {'**feature**': **time_ns**}
    :param period_ms: Window size (int ms)
    :param stride_ms: Stride size (int ms)
    :param imp_split: The impersonation type (True, False)
    :param dos_type: The DoS type ('modified', 'original')
    :param subset: A list of labels of features to be used
    :return:
    """

    path, _ = get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, parameters == {}, subset)

    if os.path.exists(path):
        metrics = load_metrics(period_ms, stride_ms, imp_split, dos_type, model, parameters == {}, subset)

        if metrics == {}:
            print(path)
    else:
        X_train_mod = __create_feature_subset(X_train, subset)
        X_test_mod = __create_feature_subset(X_test, subset)

        classifier = utility.get_classifier(model, parameters, subset)
        classifier.fit(X_train_mod, y_train)

        before = time.perf_counter_ns()
        y_predict = classifier.predict(X_test_mod)
        time_model = (time.perf_counter_ns() - before) / len(X_test_mod)

        # calculate scores on test set
        metrics = get_metrics(y_test, y_predict)

        save_metrics(metrics, period_ms, stride_ms, imp_split, dos_type, model, parameters, subset)
        time_feature = 0.0

        # find sum of feature times
        for feature in feature_time_dict.keys():
            if feature in subset:
                time_feature += feature_time_dict[feature]

        save_time(
            time_model, time_feature,
            period_ms, stride_ms,
            imp_split, dos_type,
            model, parameters,
            subset)

        print(f"Saved metrics to {path}")

    return metrics


def __save_stepwise_elimination(models, X_train, y_train, X_test, y_test, max_features,
                                feature_time_dict, period_ms, stride_ms, imp_split, dos_type):
    # runs step-wise elimination on specified parameters and saves the results of each subset model combination

    # get feature labels
    labels = (list(datapoint_attributes)[2:]).copy()
    working_set = labels.copy()

    for i in range(max_features):
        # save the feature label which yields the best result when eliminated from the pool
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
                    X_test, y_test,
                    feature_time_dict,
                    period_ms, stride_ms,
                    imp_split, dos_type,
                    current_subset) for model in models.keys()}

                for future in conf.as_completed(futures):
                    score = future.result()['total'].f1

                    if score > best_score:
                        best_score = score
                        best_label = label

        # remove the feature label which yields the best result when eliminated from the pool
        del working_set[working_set.index(best_label)]
        del labels[labels.index(best_label)]


def __create_feature_subset(X, subset):
    # returns a modified copy of input list of feature values,
    # which only contains values of features in the specified subset

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


if __name__ == "__main__":
    models = {
        'bn': {},
        'mlp': {
            'activation': 'logistic',
            'alpha': 0.0001,
            'hidden_layer_sizes': (16, 3),
            'learning_rate': 'adaptive',
            'max_iter': 300,
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

    generate_results(
        windows=[100, 50, 20, 10],
        strides=[100, 50, 20, 10],
        imp_splits=[False],
        dos_types=['modified'],
        models=models,
        eliminations=4)
