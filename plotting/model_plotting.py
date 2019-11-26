import functools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import run_models
import os
from datareader_csv import load_metrics
from datapoint import datapoint_features, datapoint_attribute_descriptions
import datareader_csv
import metrics
from run_models import selected_models
import model_selection
import numpy as np

__models = selected_models


def __generate_results(windows, strides, imp_splits, dos_types, models):
    run_models.generate_validation_results(
        windows=windows,
        strides=strides,
        imp_splits=imp_splits,
        dos_types=dos_types,
        models=models,
    )


def __get_desc(imp_split, dos_type):
    """Gets a formatted description from the impersonation split and dos split"""
    imp = "split" if imp_split is True else "unsplit"
    dos = "original" if dos_type == 'original' else "modified"

    return f"({imp} impersonation, {dos} DoS)"


def __plot_elements(elements, value_func, models, xlabel="", ylabel="", title=""):
    """
    Helper method to help plot different data
    :param elements: A list of elements to plot
    :param value_func: A function that is called on each element in elements,
    in order to get the actual value to plot
    :param models: A dictionary mapping models to their parameters
    :param xlabel: The label on the x axis of the plot
    :param ylabel: The label on the y axis of the plot
    :param title: The title of the plot
    :return:
    """
    model_labels = models.keys()

    for model in model_labels:
        values = []

        # Calculate values using value_func
        for element in elements:
            values.append(value_func(element, model))

        # Create plot for current model
        plt.scatter(elements, values, label=model)
        plt.plot(elements, values)

    # Setup and show plots
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


def plot_windows(windows, imp_split=True, dos_type='modified'):
    """
    Plots the effects of different window sizes, with F1 score on the y-axis and window sizes on the x-axis
    :param windows: A list of window sizes to plot
    :param imp_split: Boolean indicating if impersonation should be split
    :param dos_type: Value indicating if original or modified DoS should be loaded.
    :return:
    """
    def value_func(window, model):
        __generate_results([window], [window], [imp_split], [dos_type], {model: __models[model]})
        metrics = load_metrics(window, window, imp_split, dos_type, model, __models[model] == {},
                               datapoint_features)
        return metrics["total"].f1

    __plot_elements(
        windows, value_func, __models,
        "Window size (ms)",
        "F1 Score",
        "Window sizes with stride size equal to windows size\n" + __get_desc(imp_split, dos_type)
    )


def plot_strides(strides, imp_split=True, dos_type='modified'):
    """
    Plots the effects of different stride sizes, with F1 score on the y-axis and stride sizes on the x-axis
    :param strides: A list of stride sizes to plot
    :param imp_split: Boolean indicating if impersonation should be split
    :param dos_type: Value indicating if original or modified DoS should be loaded.
    :return:
    """
    def value_func(stride, model):
        metrics = load_metrics(100, stride, imp_split, dos_type, model, __models[model] == {},
                               datapoint_features)
        return metrics["total"].f1

    __generate_results([100], strides, [imp_split], [dos_type], __models)
    __plot_elements(
        strides, value_func, __models,
        "Stride size (ms)",
        "F1 Score",
        "Strides with 100ms window size\n" + __get_desc(imp_split, dos_type)
    )


def plot_feature_stride_times(strides, imp_split=True, dos_type='modified'):
    """
    Plots the time taken to calculate features only (ignoring model prediction time),
    with time on the y-axis and stride sizes on the x-axis.
    :param strides: A list of stride sizes to plot
    :param imp_split: Boolean indicating if impersonation should be split
    :param dos_type: Value indicating if original or modified DoS should be loaded.
    :return:
    """
    def value_func(stride, model):
        times = datareader_csv.load_times(100, stride, imp_split, dos_type, model, __models[model] == {},
                                          datapoint_features)
        return times["feature_time"] / 1000000

    __generate_results([100], strides, [imp_split], [dos_type], __models)
    __plot_elements(
        strides, value_func, __models,
        "Stride size (ms)",
        "Time (ms)",
        "Feature calculation times with 100ms windows\n" + __get_desc(imp_split, dos_type)
    )


def plot_feature_window_times(windows, imp_split=True, dos_type='modified'):
    """
    Plots the time taken to calculate features only (ignoring model prediction time),
    with time on the y-axis and window sizes on the x-axis.
    :param windows: A list of window sizes to plot
    :param imp_split: Boolean indicating if impersonation should be split
    :param dos_type: Value indicating if original or modified DoS should be loaded.
    :return:
    """
    def value_func(window, model):
        times = datareader_csv.load_times(window, 100, imp_split, dos_type, model, __models[model] == {},
                                          datapoint_features)
        return times["feature_time"] / 1000000

    __generate_results(windows, [100], [imp_split], [dos_type], __models)
    __plot_elements(
        windows, value_func, __models,
        "Window size (ms)",
        "Time (ms)",
        "Feature calculation times with 100ms strides\n" + __get_desc(imp_split, dos_type)
    )


def plot_model_stride_times(strides, imp_split=True, dos_type='modified'):
    """
    Plots strides with time on the y-axis and stride size on the x-axis.
    :param strides: A list of stride sizes to plot
    :param imp_split: Boolean indicating if impersonation should be split
    :param dos_type: Value indicating if original or modified DoS should be loaded.
    :return:
    """
    def value_func(stride, model):
        times = datareader_csv.load_times(100, stride, imp_split, dos_type, model, __models[model] == {},
                                          datapoint_features)
        return times["model_time"] / 1000000

    __generate_results([100], strides, [imp_split], [dos_type], __models)
    __plot_elements(
        strides, value_func, __models,
        "Stride size (ms)",
        "Time (ms)",
        "Model prediction times using 100 ms windows\n" + __get_desc(imp_split, dos_type)
    )


def plot_model_window_times(windows, imp_split=True, dos_type='modified'):
    """
    Plots windows with time on the y-axis and window size on the x-axis.
    :param windows: A list of window sizes to plot
    :param imp_split: Boolean indicating if impersonation should be split
    :param dos_type: Value indicating if original or modified DoS should be loaded.
    :return:
    """
    def value_func(window, model):
        times = datareader_csv.load_times(window, 100, imp_split, dos_type, model, __models[model] == {},
                                          datapoint_features)
        return times["model_time"] / 1000000

    __generate_results(windows, [100], [imp_split], [dos_type], __models)
    __plot_elements(
        windows, value_func, __models,
        "Window size (ms)",
        "Time (ms)",
        "Model prediction times using 100ms stride\n" + __get_desc(imp_split, dos_type)
    )


def plot_features_f1s(results, feature_labels, n_rows=1, n_columns=1, f1_type='macro', plot_type='include'):

    def get_subplot(fig, feature, index):
        ax: Axes3D = fig.add_subplot(n_rows, n_columns, index + 1)

        feature_results = results
        if plot_type == 'include':
            feature_results = metrics.filter_results(results, features=[feature])
        elif plot_type == 'exclude':
            feature_results = metrics.filter_results(results, without_features=[feature])
        else:
            raise ValueError("Invalid plot_type parameter")

        points = {}
        for result in feature_results:
            label = len(result.subset)
            y = result.metrics[f1_type].f1
            x = result.times['feature_time'] / 1e6

            points.setdefault(label, [])
            points[label].append((x, y))

        labels = list(points.keys())
        labels.sort()
        labels.reverse()
        for label in labels:
            x = [point[0] for point in points[label]]
            y = [point[1] for point in points[label]]

            ax.scatter(x, y, label=label, s=5)

        ax.legend(loc='lower right')
        ax.set_title(datapoint_attribute_descriptions[feature])

        return ax

    fig = plt.figure(figsize=(6.4 * n_columns, 4.8 * n_rows))

    for i, feature in enumerate(feature_labels):
        get_subplot(fig, feature, i)

    plt.show()


def plot_all_results_3d(results, angle=0, models=__models.keys(), windows=None, strides=None, labeling='model',
                     f1_type='weighted', title=None):
    """
    Loads all currently calculated results and plots them based on F1 score, model time and feature time.
    :param angle: The angle of the plot
    :param models: A list of model labels to plot. Pass None to use all models
    :param windows: A list of window sizes to plot. Pass None to use all sizes
    :param strides: A list of stride sizes to plot. Pass None to use all sizes
    :param labeling: A string to indicate how to label the datapoints. Valid values are:
                     'model', 'window', 'stride, 'feature_count'
    :param f1_type: A string indicating what type of f1 score to use. Valid values are: 'weighted', 'macro'
    :return:
    """

    def get_subplot(fig, pos, angle):
        ax: Axes3D = fig.add_subplot(pos, projection='3d')

        # Dict associating labels with points
        points = {}
        for result in results:
            # Get label based on labeling parameter
            label = {
                'model': result.model,
                'window': result.period_ms,
                'stride': result.stride_ms,
                'feature_count': len(result.subset),
                'dos_type': result.dos_type
            }[labeling]

            # Store point in dictionary based on label
            x = result.times["feature_time"] / 1000000
            y = result.times["model_time"] / 1000000
            z = result.metrics[f1_type].f1
            point = (x, y, z)
            points.setdefault(label, [])
            points[label].append(point)

        for label in points.keys():
            x = [point[0] for point in points[label]]
            y = [point[1] for point in points[label]]
            z = [point[2] for point in points[label]]
            ax.scatter(x, y, z, label=label, s=5)

        # Set plot limits
        ax.set_ylim3d(0, 0.3)
        ax.set_zlim3d(0.6, 1)

        # Setup and show plots
        ax.set_xlabel("Feature time (ms)")
        ax.set_ylabel("Model time (ms)")
        ax.set_zlabel("F1 score")
        ax.view_init(10, angle)

    results = metrics.filter_results(results, models=models, periods=windows, strides=strides)

    fig = plt.figure(figsize=(12.8, 4.8))
    get_subplot(fig, 121, angle)
    get_subplot(fig, 122, 90-angle)

    if title is None:
        title = "Correlation between F1 score and times\nPoint colors = " + labeling.replace("_", " ")

    plt.legend(loc='lower right')
    plt.suptitle(title)
    plt.show()


def plot_all_results_2d(results, models=__models.keys(), windows=None, strides=None, labeling='model',
                     f1_type='weighted', title=None):

    def get_subplot(fig, pos, is_model_time):
        ax = fig.add_subplot(pos)

        # Dict associating labels with points
        points = {}
        for result in results:
            # Get label based on labeling parameter
            label = {
                'model': result.model,
                'window': result.period_ms,
                'stride': result.stride_ms,
                'feature_count': len(result.subset),
                'dos_type': result.dos_type
            }[labeling]

            # Store point in dictionary based on label
            if is_model_time:
                x = result.times["model_time"] / 1000000
            else:
                x = result.times["feature_time"] / 1000000

            y = result.metrics[f1_type].f1
            point = (x, y)
            points.setdefault(label, [])
            points[label].append(point)

        for label in points.keys():
            x = [point[0] for point in points[label]]
            y = [point[1] for point in points[label]]
            ax.scatter(x, y, label=label, s=5)

        # Setup and show plots
        if is_model_time:
            ax.set_xlabel("Model time (ms)")
        else:
            ax.set_xlabel("Feature time (ms)")
        ax.set_ylabel("F1 score")

        if is_model_time:
            ax.set_title("\nModel time")
        else:
            ax.set_title("\nFeature time")

    results = metrics.filter_results(results, models=models, periods=windows, strides=strides)

    fig = plt.figure(figsize=(12.8, 4.8))
    get_subplot(fig, 121, True)
    get_subplot(fig, 122, False)

    if title is None:
        title = "Correlation between F1 score and times\nPoint colors = " + labeling.replace("_", " ")

    plt.legend(loc='lower right')
    plt.suptitle(title)
    plt.show()


def plot_barchart_results(results, plot_type='f1_macro'):
    ys = []
    models = []

    y_func, title = {
        'f1_macro':         (lambda r: r.metrics['macro'].f1,         "F1 macro average"),
        'f1_weighted':      (lambda r: r.metrics['weighted'].f1,      "F1 weighted average"),
        'f1_normal':        (lambda r: r.metrics['normal'].f1,        "F1 normal"),
        'f1_impersonation': (lambda r: r.metrics['impersonation'].f1, "F1 impersonation"),
        'f1_dos':           (lambda r: r.metrics['dos'].f1,           "F1 DoS"),
        'f1_fuzzy':         (lambda r: r.metrics['fuzzy'].f1,         "F1 fuzzy"),
        'model_time':       (lambda r: r.times['model_time'] / 1e6,   "Model prediction time (ms)"),
        'feature_time':     (lambda r: r.times['feature_time'] / 1e6, "Feature calculation time (ms)"),
    }[plot_type]

    for result in results:
        ys.append(y_func(result))
        models.append(result.model)

    plt.bar(models, ys)
    plt.title(title)
    plt.show()


def plot_barchart_feature_results(results):
    bars = {}  # List of tuples with: (index, y value, bottom, label)
    for i, result in enumerate(results):
        feature_durations = metrics.get_result_feature_breakdown(result)

        current_bottom = 0
        for feature in result.subset:
            duration = feature_durations[feature] / 1e6
            bars.setdefault(feature, [])
            bars[feature].append((result.model, duration, current_bottom, feature))
            current_bottom += duration

    for feature, bars in bars.items():
        ind = [bar[0] for bar in bars]
        durations = [bar[1] for bar in bars]
        bottoms = [bar[2] for bar in bars]

        plt.bar(ind, durations, bottom=bottoms, align='center', label=datapoint_attribute_descriptions[feature])

    plt.title("Feature time breakdown of selected models (ms)")
    plt.legend(fontsize="small")
    plt.show()


def plot_feature_barcharts(times_dict):
    features = []
    times = []
    for feature, time in times_dict.items():
        times.append(time / 1e6)
        features.append(datapoint_attribute_descriptions[feature])

    plt.bar(features, times)
    plt.xticks(rotation=-80)
    plt.title("Average feature calculation times.")
    plt.show()


def plot_barchart_subsets(results: [metrics.Result], labels=None, f1_type='macro'):
    def subset_hash(subset):
        return hash(functools.reduce(lambda a, b: a + "," + str(b), subset))

    # Sets of seen models and subset
    models = set()
    subsets = set()

    # Organize models and scores
    scores = {}
    for result in results:
        model = result.model
        subset = subset_hash(result.subset)
        models.add(model)
        subsets.add(subset)

        scores.setdefault(model, {})
        scores[model][subset] = result.metrics[f1_type].f1

    bar_width = 0.1
    origins = np.arange(len(models))

    # Create set of bars for each subset
    for i, subset in enumerate(subsets):
        positions = [ind + i * bar_width for ind in origins]
        subset_scores = [scores[model].get(subset, 0) for model in models]

        label = labels[i] if labels is not None else i
        plt.bar(positions, subset_scores, width=bar_width, label=label)

    plt.ylabel("F1 Score")
    plt.xticks(origins, list(models))
    plt.legend()
    plt.show()


def __get_in_range(xs, ys, min_x, max_x):
    xs_new = []
    ys_new = []

    for x, y in zip(xs, ys):
        if min_x <= x <= max_x:
            xs_new.append(x)
            ys_new.append(y)

    return xs_new, ys_new


def plot_transition_dataset(results, model_labels):
    best_results = model_selection.get_best_for_models(results, model_labels, 0, 0, 1)

    transition = 0
    max_timestamp = 0
    min_timestamp = None

    for configuration in best_results:
        if configuration.model == 'bn':
            continue

        probabilities, timestamps, transition = run_models.get_transition_class_probabilities(configuration)

        timestamps, probabilities = __get_in_range(timestamps, probabilities, transition - 1000, transition + 1000)
        plt.plot(timestamps, probabilities, label=configuration.model)

        if min_timestamp is None:
            min_timestamp = timestamps[0]

        max_timestamp = max([max_timestamp] + timestamps)
        min_timestamp = min([min_timestamp] + timestamps)

    plt.plot([min_timestamp, transition, transition, max_timestamp], [0, 0, 1, 1], label="Ground truth")
    plt.plot([min_timestamp, max_timestamp], [0.5, 0.5], label="Threshold")

    plt.legend()
    plt.ylabel("Impersonation probability")
    plt.xlabel("Time (ms)")
    plt.show()


if __name__ == '__main__':
    os.chdir("..")

    # Options
    dos_type = 'original'  # 'modified' or 'original'
    imp_type = 'imp_split'  # 'imp_split' or 'imp_full'

    _models = list(__models.keys())
    del _models[0]

    results = datareader_csv.load_all_results()
    validation_results = metrics.filter_results(results, dos_types=[dos_type], is_test=False)
    test_results = metrics.filter_results(results, dos_types=[dos_type], is_test=True)

    plot_barchart_subsets(validation_results, ["A", "B", "C", "D"])

    #plot_transition_dataset(validation_results, _models)

    # Bar plots
    bar_types = ['f1_macro', 'f1_weighted', 'f1_normal', 'f1_impersonation', 'f1_dos', 'f1_fuzzy', 'model_time',
                 'feature_time']
    for type in bar_types:
        plot_barchart_results(test_results, type)
    plot_barchart_feature_results(test_results)

    durations_path = f"data\\feature\\{imp_type}\\{dos_type}\\mixed_validation_time_100ms_100ms.csv"
    feature_times = datareader_csv.load_feature_durations(durations_path)
    del feature_times['time_ms']
    del feature_times['class_label']
    plot_feature_barcharts(feature_times)

    feature_results = metrics.filter_results(validation_results)
    plot_features_f1s(feature_results, datapoint_features[0:1], 1, 1)
    plot_features_f1s(feature_results, datapoint_features[1:], 3, 3)

    plot_features_f1s(feature_results, datapoint_features, 5, 2, plot_type='include')
    plot_features_f1s(feature_results, datapoint_features, 5, 2, plot_type='exclude')

    # Options
    _models = __models.keys()
    _windows = [100, 50, 20, 10]
    _strides = [100, 50, 20, 10]
    angle = 15

    labelings = ['model', 'window', 'stride', 'feature_count', 'dos_type']
    f1_type = 'macro'

    for labeling in labelings:
        plot_all_results_2d(validation_results, _models, _windows, _strides, labeling, f1_type)
        plot_all_results_3d(validation_results, angle, _models, _windows, _strides, labeling, f1_type)


    #plot_windows(_windows, imp_split=False, dos_type='modified')
    #plot_strides(_strides, imp_split=False, dos_type='modified')
    #plot_feature_stride_times(_strides, imp_split=False, dos_type='modified')
    #plot_feature_window_times(_windows, imp_split=False, dos_type='modified')
    #plot_model_window_times(_windows, imp_split=False, dos_type='modified')
    #plot_model_stride_times(_strides, imp_split=False, dos_type='modified')
