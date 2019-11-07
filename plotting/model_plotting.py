from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import run_models
import os
from datareader_csv import load_metrics
from datapoint import datapoint_attributes
import datareader_csv
import metrics
from run_models import selected_models

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
                               list(datapoint_attributes)[2:])
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
                               list(datapoint_attributes)[2:])
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
                                          list(datapoint_attributes)[2:])
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
                                          list(datapoint_attributes)[2:])
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
                                          list(datapoint_attributes)[2:])
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
                                          list(datapoint_attributes)[2:])
        return times["model_time"] / 1000000

    __generate_results(windows, [100], [imp_split], [dos_type], __models)
    __plot_elements(
        windows, value_func, __models,
        "Window size (ms)",
        "Time (ms)",
        "Model prediction times using 100ms stride\n" + __get_desc(imp_split, dos_type)
    )


def plot_all_results(angle=0, models=__models.keys(), windows=None, strides=None, labeling='model'):
    """
    Loads all currently calculated results and plots them based on F1 score, model time and feature time.
    :param angle: The angle of the plot
    :param models: A list of model labels to plot. Pass None to use all models
    :param windows: A list of window sizes to plot. Pass None to use all sizes
    :param strides: A list of stride sizes to plot. Pass None to use all sizes
    :param labeling: A string to indicate how to label the datapoints. Valid values are:
                     'model', 'window', 'stride, 'feature_count'
    :return:
    """
    results = datareader_csv.load_all_results()
    results = metrics.filter_results(results, models=models, periods=windows, strides=strides, dos_types='modified')

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')

    # Dict associating labels with points
    points = {}

    for result in results:
        # Get label based on labeling parameter
        label = {
            'model': result.model,
            'window': result.period_ms,
            'stride': result.stride_ms,
            'feature_count': len(result.subset)
        }[labeling]

        # Store point in dictionary based on label
        point = (
            result.times["feature_time"] / 1000000,
            result.times["model_time"] / 1000000,
            result.metrics["total"].f1
        )
        points.setdefault(label, [])
        points[label].append(point)

    for label in points.keys():
        x = [point[0] for point in points[label]]
        y = [point[1] for point in points[label]]
        z = [point[2] for point in points[label]]
        ax.scatter(x, y, z, label=label, s=10)

    # Set plot limits
    ax.set_ylim3d(0, 0.3)
    ax.set_zlim3d(0.6, 1)

    # Setup and show plots
    ax.set_xlabel("Feature time (ms)")
    ax.set_ylabel("Model time (ms)")
    ax.set_zlabel("F1 score")
    ax.view_init(20, angle)
    plt.legend(loc='lower right')
    plt.title("Correlation between F1 score and times \nPoint colors = " + labeling.replace("_", " "))
    plt.show()


if __name__ == '__main__':
    os.chdir("..")

    _models = __models.keys()
    _windows = [100, 50, 10]
    _strides = [100, 50, 10]
    angle1 = 10
    angle2 = 80

    labelings = ['model', 'window', 'stride', 'feature_count']

    for labeling in labelings:
        plot_all_results(angle1, _models, _windows, _strides, labeling)
        plot_all_results(angle2, _models, _windows, _strides, labeling)

    #plot_windows(_windows, imp_split=False, dos_type='modified')
    #plot_strides(_strides, imp_split=False, dos_type='modified')
    #plot_feature_stride_times(_strides, imp_split=False, dos_type='modified')
    #plot_feature_window_times(_windows, imp_split=False, dos_type='modified')
    #plot_model_window_times(_windows, imp_split=False, dos_type='modified')
    #plot_model_stride_times(_strides, imp_split=False, dos_type='modified')
