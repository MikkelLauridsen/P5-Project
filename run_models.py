"""Functions for generating results by running the models with different settings."""
import os
import time
import concurrent.futures as conf

from sklearn.calibration import CalibratedClassifierCV

import models.model_utility as utility
import hugin.pyhugin87 as hugin
from sklearn.preprocessing import StandardScaler
from models.model_utility import get_training_validation
from metrics import get_metrics, get_metrics_path, get_error_metrics
from datapoint import datapoint_features
from datareader_csv import load_metrics
from datawriter_csv import save_metrics, save_time
from datasets import get_transitioning_dataset
import configuration as config


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
                    X_train, y_train, X_validation, y_validation, feature_time_dict = get_training_validation(
                        period_ms, stride_ms,
                        imp_split, dos_type)

                    print(f"starting jobs {current_job} through {current_job + inner_loop_size} of "
                          f"{job_count} -- {(current_job / job_count) * 100.0}%")

                    with conf.ProcessPoolExecutor() as executor:
                        for model in models.keys():
                            if eliminations > 0:
                                executor.submit(
                                    __save_backward_elimination,
                                    model, models[model],
                                    X_train, y_train,
                                    X_validation, y_validation,
                                    eliminations,
                                    feature_time_dict,
                                    period_ms, stride_ms,
                                    imp_split, dos_type)
                            else:
                                executor.submit(
                                    create_and_save_results,
                                    model, models[model],
                                    X_train, y_train,
                                    X_validation, y_validation,
                                    feature_time_dict,
                                    period_ms, stride_ms,
                                    imp_split, dos_type,
                                    datapoint_features)

                    current_job += inner_loop_size


def __get_stepwise_size(max_features):
    # Returns the number of feature subsets to be tested.

    n = len(datapoint_features)
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


def __save_backward_elimination(model, parameters, X_train, y_train, X_validation, y_validation, max_features,
                                feature_time_dict, period_ms, stride_ms, imp_split, dos_type):
    # Runs step-wise elimination on specified parameters and saves the results of each subset model combination.

    # Get feature labels.
    working_set = datapoint_features.copy()

    for _ in range(max_features):
        # Save the feature label which yields the best result when eliminated from the pool.
        best_score = 0
        best_label_index = 0

        for i in range(len(working_set)):
            current_subset = working_set.copy()
            del current_subset[i]

            metrics = create_and_save_results(
                model, parameters,
                X_train, y_train,
                X_validation, y_validation,
                feature_time_dict,
                period_ms, stride_ms,
                imp_split, dos_type,
                current_subset)

            score = metrics['macro'].f1

            if score > best_score:
                best_score = score
                best_label_index = i

        # Remove the feature label which yields the best result when eliminated from the pool.
        del working_set[best_label_index]


def __create_feature_subset(X, subset):
    # Returns a modified copy of input list of feature values,
    # which only contains values of features in the specified subset.

    indices = [datapoint_features.index(f) for f in subset]
    length = len(datapoint_features)

    X_mod = []

    for sample in X:
        sample_mod = []

        for i in range(0, length):
            if i in indices:
                sample_mod.append(sample[i])

        X_mod.append(sample_mod)

    return X_mod


def get_transition_class_probabilities(configuration, run_stride):
    # Create dataset to train on
    X_train, y_train, X_validation, y_validation, _ = utility.get_training_validation(
        configuration.period_ms, configuration.stride_ms,
        configuration.imp_split, configuration.dos_type, False
    )
    X_train = __create_feature_subset(list(X_train) + list(X_validation), configuration.subset)
    y_train = list(y_train) + list(y_validation)

    # Create dataset to predict on
    dataset, transition = get_transitioning_dataset(configuration.period_ms, run_stride, verbose=True)
    X, _ = utility.split_feature_label(dataset)
    X = __create_feature_subset(X, configuration.subset)

    # Scale datasets
    scaler = utility.get_scaler(X_train)
    X_train = scaler.transform(X_train)
    X = scaler.transform(X)

    # Create and fit classifier
    classifier = utility.get_classifier(configuration.model, config.selected_models[configuration.model], configuration.subset)
    if configuration.model != 'bn':
        classifier = CalibratedClassifierCV(classifier, cv=5)
    classifier.fit(X_train, y_train)

    predictions = list(classifier.predict_proba(X))
    timestamps = [window.time_ms for window in dataset]
    imp_predictions = []

    imp_index = list(classifier.classes_).index('impersonation')

    for prediction in predictions:
        imp_predictions.append(prediction[imp_index])

    return imp_predictions, timestamps, transition


if __name__ == "__main__":
    # Generate for dataset. Use configuration.py to specify type
    generate_validation_results(
        windows=[100, 50, 20, 10],
        strides=[10, 20, 50, 100],
        imp_splits=[config.imp_split],
        dos_types=[config.dos_type],
        models=config.selected_models,
        eliminations=4)
