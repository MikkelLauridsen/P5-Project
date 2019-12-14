import functools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import run_models
import os
from datareader_csv import load_metrics
from datapoint import datapoint_features, datapoint_attribute_descriptions, index_to_feature_label
import datareader_csv
import metrics
import model_selection
import numpy as np
import configuration as conf
from datasets import get_transitioning_dataset

__models = conf.selected_models


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
    """
    Plots performance of results, where a specific feature is either included or excluded
    :param results: Results to plot
    :param feature_labels: Feature labels to generate plots from
    :param n_rows: Number of rows determining how to setup the plots
    :param n_columns: Number of columns determining how to setup the plots
    :param f1_type: F1 scoring type
    :param plot_type: 'include' or 'exclude'. Determines what plot to make
    :return:
    """
    def get_subplot(fig, feature, index):
        ax: Axes3D = fig.add_subplot(n_rows, n_columns, index + 1)

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
    Loads all currently calculated results and creates a 3d plot based on F1 score, model time and feature time.
    :param results: Results to plot
    :param angle: The angle of the plot
    :param models: A list of model labels to plot. Pass None to use all models
    :param windows: A list of window sizes to plot. Pass None to use all sizes
    :param strides: A list of stride sizes to plot. Pass None to use all sizes
    :param labeling: A string to indicate how to label the datapoints. Valid values are:
                     'model', 'window', 'stride, 'feature_count'
    :param f1_type: A string indicating what type of f1 score to use. Valid values are: 'weighted', 'macro'
    :param title: Title of plot
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
                'window': result.window_ms,
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

    results = metrics.filter_results(results, models=models, windows=windows, strides=strides)

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
    """
    Loads all currently calculated results and plots them based on F1 score, model time and feature time.
    :param results: Results to plot
    :param models: A list of model labels to plot. Pass None to use all models
    :param windows: A list of window sizes to plot. Pass None to use all sizes
    :param strides: A list of stride sizes to plot. Pass None to use all sizes
    :param labeling: A string to indicate how to label the datapoints. Valid values are:
                     'model', 'window', 'stride, 'feature_count'
    :param f1_type: A string indicating what type of f1 score to use. Valid values are: 'weighted', 'macro'
    :param title: Title of plot
    :return:
    """

    def get_subplot(fig, pos, is_model_time):
        ax = fig.add_subplot(pos)

        # Dict associating labels with points
        points = {}
        for result in results:
            # Get label based on labeling parameter
            label = {
                'model': result.model,
                'window': result.window_ms,
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

    results = metrics.filter_results(results, models=models, windows=windows, strides=strides)

    # Create both plots
    fig = plt.figure(figsize=(12.8, 4.8))
    get_subplot(fig, 121, True)
    get_subplot(fig, 122, False)

    if title is None:
        title = "Correlation between F1 score and times\nPoint colors = " + labeling.replace("_", " ")

    plt.legend(loc='lower right')
    plt.suptitle(title)
    plt.show()


def plot_barchart_results(results, plot_type='f1', metrics_type='macro'):
    """
    Creates various barchart plots for results.
    :param results: Results to plot
    :param plot_type: What type of information to plot with bars. Valid values are:
        'f1', 'fpr', 'fnr', 'recall', 'precision', 'accuracy', 'model_time' and 'feature_time'
    :param metrics_type: What metric type to use
    :return:
    """
    ys = []
    models = []

    metrics_name = metrics_type
    if metrics_name == 'normal':
        metrics_name = "attack-free"

    # Determine y function and title from plot type
    y_func, title = {
        'f1':           (lambda r: r.metrics[metrics_type].f1,                f"F1 {metrics_name}"),
        'fpr':          (lambda r: r.metrics[metrics_type].fpr,               f"FPR {metrics_name}"),
        'fnr':          (lambda r: r.metrics[metrics_type].fnr,               f"FNR {metrics_name}"),
        'recall':       (lambda r: r.metrics[metrics_type].recall,            f"Recall {metrics_name}"),
        'precision':    (lambda r: r.metrics[metrics_type].precision,         f"Precision {metrics_name}"),
        'accuracy':     (lambda r: r.metrics[metrics_type].balanced_accuracy, f"Accuracy {metrics_name}"),
        'model_time':   (lambda r: r.times['model_time'] / 1e6,                "Model prediction time (ms)"),
        'feature_time': (lambda r: r.times['feature_time'] / 1e6,              "Feature calculation time (ms)"),
    }[plot_type]

    for result in results:
        ys.append(y_func(result))
        models.append(result.model)

    plt.ylim(0, 1)
    plt.bar(models, ys)
    plt.title(title)
    plt.savefig(f"plots/{plot_type}_{metrics_type}")
    plt.show()


def plot_barchart_feature_results(results):
    """
    Creates a barchart showing a breakdown of what features take time to calculate for results
    :param results: Results to plot
    :return:
    """
    bars = {}  # List of tuples with: (index, y value, bottom, label)
    for i, result in enumerate(results):
        feature_durations = metrics.get_result_feature_breakdown(result)

        # Add a bar for each feature in the result, keeping track it height and bottom height
        current_bottom = 0
        for feature in result.subset:
            duration = feature_durations[feature] / 1e6
            bars.setdefault(feature, [])
            bars[feature].append((result.model, duration, current_bottom, feature))
            current_bottom += duration

    # Create bars for each feature
    for feature, bars in bars.items():
        ind = [bar[0] for bar in bars]
        durations = [bar[1] for bar in bars]
        bottoms = [bar[2] for bar in bars]

        plt.bar(ind, durations, bottom=bottoms, align='center', label=datapoint_attribute_descriptions[feature])

    plt.title("Feature time breakdown of selected models (ms)")
    plt.legend(fontsize="small")
    plt.savefig("plots/feature_times_colored")
    plt.show()


def plot_feature_barcharts(times_dict):
    """
    Creates a barchart showinghwo long featues take to calculate
    :param times_dict: Dictionary mapping features to times
    :return:
    """
    features = []
    times = []
    for feature, time in times_dict.items():
        times.append(time / 1e6)
        features.append(datapoint_attribute_descriptions[feature])

    plt.bar(features, times)
    plt.xticks(rotation=-80)
    plt.title("Average feature calculation times.")
    plt.show()


def plot_barchart_subsets(results, models=None, subsets=None, labels=None, title="", bar_width=0.2, f1_type='macro'):
    """
    Plots performance of results based on their feature subsets
    :param results: Results to include in plot
    :param models: Model labels to plot
    :param subsets: Feature subsets to include
    :param labels: Labels to use on plot legend
    :param title: Title of plot
    :param bar_width: Width of bars
    :param f1_type: What metric type to use
    :return:
    """

    # Since sets are hashable, use a simple function to hash them in order to use them as keys in dictionaries
    def subset_hash(subset):
        return hash(functools.reduce(lambda a, b: a + "," + str(b), subset))

    known_models = set()
    known_subsets = set()
    # Find known_models if models are unspecified
    if models is None:
        for result in results:
            model = result.model
            known_models.add(model)
    else:
        known_models = models

    # Find known_subsets if subsets are unspecified
    if subsets is None:
        for result in results:
            subset = subset_hash(result.subset)
            known_subsets.add(subset)
    else:
        known_subsets = [subset_hash(subset) for subset in subsets]

    # Organize models and scores
    scores = {}
    for result in results:
        scores.setdefault(result.model, {})
        scores[result.model][subset_hash(result.subset)] = result.metrics[f1_type].f1

    origins = np.arange(len(known_models))

    # Create set of bars for each subset
    for i, subset in enumerate(known_subsets):
        positions = [ind + i * bar_width for ind in origins]
        subset_scores = [scores[model].get(subset, 0) for model in known_models]

        label = labels[i] if labels is not None else "Subset " + str(i)
        plt.bar(positions, subset_scores, width=bar_width, label=label)

    plt.ylabel("F1 Score")
    plt.xticks([pos + bar_width*len(known_subsets)/2 - bar_width/2 for pos in origins], list(known_models))
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


# Gets the sublist of a list where values outside a specific range are removed
def __get_in_range(xs, ys, min_x, max_x):
    xs_new = []
    ys_new = []

    for x, y in zip(xs, ys):
        if min_x <= x <= max_x:
            xs_new.append(x)
            ys_new.append(y)

    return xs_new, ys_new


def plot_transition_dataset(results, model_labels, run_stride=5, slice_sizes=[250, 250], include_predictions=True, weights=(0, 0, 1)):
    """
    Trains models on the test dataset and plots their performance on an artificial dataset that transitions
        from normal state to impersonation attack state.
    :param results: The results for which to choose the best models from
    :param model_labels: Model labels to include in plot
    :param run_stride: The stride of the models. I.e. how long (in ms) of a duration there should be between predictions
    :param slice_sizes: Sizes of slices to combine into one dataset
    :param include_predictions: Whether or not to include predictions in plot
    :param weights: Weights to find best results. Must be a 3-tuple
    :return:
    """

    # Find best result for each model
    best_results = model_selection.get_best_for_models(results, model_labels, *weights, 'normal')

    transitions = []
    max_timestamp = 0
    min_timestamp = None

    plt.figure(figsize=[6.4*1.5, 4.8])

    legend_groups = {}
    for configuration in best_results:
        dataset, transitions = get_transitioning_dataset(configuration.window_ms, run_stride, slice_sizes, True)

        # Train models and get their probabilities on the transition dataset
        timestamps, probabilities, predictions = run_models.get_impersonation_probabilities(configuration, dataset)

        # Remove probabilities outside range from transition point
        offset = timestamps[0]

        # Offset points to start from 0
        for i in range(len(transitions)):
            transitions[i] -= offset

        for i in range(len(timestamps)):
            timestamps[i] -= offset

        # Plot model
        plt.plot(timestamps, probabilities, label=configuration.model)

        for label in ['normal', 'dos', 'fuzzy', 'impersonation']:
            filtered_items = list(filter(lambda x: x[2] == label, list(zip(timestamps, probabilities, predictions))))
            legend_groups.setdefault(label, [])
            legend_groups[label] += filtered_items

        if min_timestamp is None:
            min_timestamp = timestamps[0]

        max_timestamp = max([max_timestamp] + timestamps)
        min_timestamp = min([min_timestamp] + timestamps)

    if include_predictions:
        for label in ['normal', 'dos', 'fuzzy', 'impersonation']:
            if len(legend_groups[label]) == 0:
                continue

            color = {
                'normal': '#C3C3C3',
                'impersonation': '#585858',
            }.get(label, None)

            xs = list(map(lambda x: x[0], legend_groups[label]))
            ys = list(map(lambda x: x[1], legend_groups[label]))
            plt.scatter(xs, ys, label=label, s=25, color=color)


    # Create ground truth line
    transition_xs = [min_timestamp]
    for transition in transitions:
        transition_xs += [transition, transition]
    transition_xs.append(max_timestamp)

    transition_ys = [0]
    for i in range(len(transitions)):
        transition_ys += [0, 1] if i % 2 == 0 else [1, 0]
    transition_ys.append(0 if len(transitions) % 2 == 0 else 1)
    # Plot ground truth line
    plt.plot(transition_xs, transition_ys, label="Ground truth", linewidth=3)

    plt.title("Normal to impersonation transition probabilities")
    plt.legend()
    plt.ylabel("Impersonation probability")
    plt.xlabel("Time (ms)")
    plt.show()


if __name__ == '__main__':
    os.chdir("..")

    _models = __models.keys()

    results = datareader_csv.load_all_results()
    validation_results = metrics.filter_results(results, dos_types=[conf.dos_type], is_test=False)
    test_results = metrics.filter_results(results, dos_types=[conf.dos_type], is_test=True)

    # _models = ['rf']
    for weights in [(0, 0, 1), (-1, -1, 0)]:
        plot_transition_dataset(validation_results, _models, 5, [300, 400, 300], True, weights)
        plot_transition_dataset(validation_results, _models, 5, [300, 50, 50, 50, 50, 50, 250], True, weights)

    # Subset plotting stuff
    barchart_subsets_results = metrics.filter_results(validation_results, [100])
    subset1 = [index_to_feature_label(index) for index in [1, 10, 11]]
    subset2 = [index_to_feature_label(index) for index in [0, 5, 9, 12, 14]]
    subset3 = [index_to_feature_label(index) for index in [6, 7, 8, 13]]
    subsets = [subset1, subset2, subset3]
    labels = ["Message frequency", "Message interval", "Message data-field"]
    title = f"Relation between performance and feature groups with 100ms windows"
    plot_barchart_subsets(barchart_subsets_results, None, subsets, labels, title)

    # Bar plots
    best_test_results = model_selection.get_best_for_models(test_results, conf.selected_models.keys(), 0, 0, 1, 'normal', True)
    plot_types = ['f1', 'fpr', 'fnr', 'recall', 'precision', 'accuracy']
    metrics_types = ['macro', 'normal', 'impersonation', 'dos', 'fuzzy']

    for plot_type in plot_types:
        for metrics_type in metrics_types:
            plot_barchart_results(best_test_results, plot_type, metrics_type)
    plot_barchart_results(best_test_results, 'model_time')
    plot_barchart_feature_results(best_test_results)

    durations_path = f"data\\feature\\{conf.imp_type}\\{conf.dos_type}\\mixed_validation_time_100ms_100ms.csv"
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
