import matplotlib.pyplot as plt
import run_models
import os
from datareader_csv import load_metrics
from datapoint import datapoint_attributes
import datareader_csv

__models = {
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


def __generate_results(windows, strides, imp_splits, dos_types, models):
    run_models.generate_results(
        windows=windows,
        strides=strides,
        imp_splits=imp_splits,
        dos_types=dos_types,
        models=models,
    )


def __plot_elements(elements, value_func, models, xlabel="", ylabel="", title=""):
    model_labels = models.keys()

    for model in model_labels:
        values = []

        for element in elements:
            values.append(value_func(element, model))

        plt.scatter(elements, values, label=model)
        plt.plot(elements, values)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


def plot_windows(windows, imp_split=True, dos_type='modified'):
    def value_func(window, model):
        __generate_results([window], [window], [imp_split], [dos_type], {model: __models[model]})
        metrics = load_metrics(window, window, imp_split, dos_type, model, __models[model],
                               list(datapoint_attributes)[2:])
        return metrics["total"].f1

    __plot_elements(
        windows, value_func, __models,
        "Window size (ms)",
        "F1 Score",
        "Window sizes with stride size equal to windows size"
    )


def plot_strides(strides, imp_split=True, dos_type='modified'):
    def value_func(stride, model):

        metrics = load_metrics(100, stride, imp_split, dos_type, model, __models[model],
                               list(datapoint_attributes)[2:])
        return metrics["total"].f1

    __generate_results([100], strides, [imp_split], [dos_type], __models)
    __plot_elements(
        strides, value_func, __models,
        "Stride size (ms)",
        "F1 Score",
        "Strides with 100ms window size"
    )


def plot_feature_stride_times(strides, imp_split=True, dos_type='modified'):
    def value_func(stride, model):
        times = datareader_csv.load_times(100, stride, imp_split, dos_type, model, __models[model],
                                          list(datapoint_attributes)[2:])
        return float(times["feature_time"]) / 1000000

    __generate_results([100], strides, [imp_split], [dos_type], __models)
    __plot_elements(
        strides, value_func, __models,
        "Stride size (ms)",
        "Time (ms)",
        "Feature calculation times with 100ms windows"
    )


def plot_feature_window_times(windows, imp_split=True, dos_type='modified'):
    def value_func(window, model):
        times = datareader_csv.load_times(window, 100, imp_split, dos_type, model, __models[model],
                                          list(datapoint_attributes)[2:])
        return float(times["feature_time"]) / 1000000

    __generate_results(windows, [100], [imp_split], [dos_type], __models)
    __plot_elements(
        windows, value_func, __models,
        "Window size (ms)",
        "Time (ms)",
        "Feature calculation times with 100ms strides"
    )


def plot_model_stride_times(strides, imp_split=True, dos_type='modified'):
    def value_func(stride, model):
        times = datareader_csv.load_times(100, stride, imp_split, dos_type, model, __models[model],
                                          list(datapoint_attributes)[2:])
        return float(times["model_time"]) / 1000000

    __generate_results([100], strides, [imp_split], [dos_type], __models)
    __plot_elements(
        strides, value_func, __models,
        "Stride size (ms)",
        "Time (ms)",
        "Model prediction times using 100 ms windows"
    )


def plot_model_window_times(windows, imp_split=True, dos_type='modified'):
    def value_func(window, model):
        times = datareader_csv.load_times(window, 100, imp_split, dos_type, model, __models[model],
                                          list(datapoint_attributes)[2:])
        return float(times["model_time"]) / 1000000

    __generate_results(windows, [100], [imp_split], [dos_type], __models)
    __plot_elements(
        windows, value_func, __models,
        "Window size (ms)",
        "Time (ms)",
        "Model prediction times using 100ms stride"
    )


if __name__ == '__main__':
    os.chdir("..")

    _windows = [10, 25, 50, 100]
    _strides = [200, 100, 50, 25, 10]

    plot_windows(_windows)
    plot_strides(_strides)
    plot_feature_stride_times(_strides)
    plot_feature_window_times(_windows)
    plot_model_window_times(_windows)
    plot_model_stride_times(_strides)
