import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import datasets as ds
import datapoint
import datareader_csv
import operator
import configuration as conf

def plot_feature_barcharts(times_dict):
    features = []
    times = []

    # Sorting the times dict by the value.
    sorted_times_dict = sorted(times_dict.items(), key=operator.itemgetter(1))

    sorted_times_dict.reverse()

    for feature, time in sorted_times_dict:
        times.append(time / 1e6)
        features.append(datapoint.datapoint_attribute_descriptions[feature])

    plt.barh(features, times)
    plt.xticks(fontsize=7.5)
    plt.title("Average feature calculation times")
    plt.ylabel("Time (ms)")
    plt.show()


def __class_to_color(cls):
    # Gets a color from a class label
    return {
        "normal": "#246EB6",
        "dos": "#B62121",
        "fuzzy": "#0A813E",
        "impersonation": "#FF7A0E"
    }.get(cls, "#000000")


def plot_all_features(datapoints):
    """
    Plots all features from the DataPoint class with
    the feature value on the y-axis and time (of datapoint) on the x-axis
    :param datapoints: Datapoints to plot features of
    :return:
    """
    time_ms = [point.time_ms for point in datapoints]

    # Extract features from DataPoints
    class_label = [point.class_label for point in datapoints]

    for attr in datapoint.datapoint_attributes:
        feature_list = [getattr(point, attr) for point in datapoints]
        # Get attribute description if available
        feature_description = datapoint.datapoint_attribute_descriptions.get(attr, attr)

        # Ignore time_ms and class_label attributes, as these are not features
        if attr == "time_ms" or attr == "class_label":
            pass

        # Special case for mean_id_intervals_variance and req_to_res_time_variance, to manually set axis limits
        elif attr == "mean_id_intervals_variance" or attr == "req_to_res_time_variance":
            __setup_scatter(time_ms, feature_list, "Time", feature_description, class_label, False)
            plt.ylim(top=0.0005, bottom=-0.00025)  # Manually set axis limits
            plt.show()
        else:
            __setup_scatter(time_ms, feature_list, "Time", feature_description, class_label)


def __setup_scatter(xaxis, yaxis, xlabel, ylabel, class_label, show=True):
    # Helper function for setting up a scatter plot.
    # The function does not call plt.show() in order to allow further configuration afterwards.
    colors = [__class_to_color(cls) for cls in class_label]
    legends = [(__class_to_color("normal"), "attack free state"),
               (__class_to_color("dos"), "DoS attack state"),
               (__class_to_color("fuzzy"), "Fuzzy attack state"),
               (__class_to_color("impersonation"), "Impersonation attack state")]

    patches = []

    for legend in legends:
        patches += [mpatches.Patch(color=legend[0], label=legend[1])]

    plt.figure(figsize=(12, 7))
    plt.scatter(xaxis, yaxis, s=5, c=colors, label="")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles=patches)

    if show:
        plt.show()


if __name__ == "__main__":
    os.chdir("..")

    # Get datasets
    training_points, validation_points, _ = ds.load_or_create_datasets(
        window_ms=100,
        stride_ms=100,
        imp_split=conf.imp_split,
        dos_type=conf.dos_type,
        in_parallel=True)

    # Plot features
    plot_all_features(training_points)

    durations_path = f"data\\feature\\{conf.imp_type}\\{conf.dos_type}\\mixed_validation_time_100ms_100ms.csv"
    feature_times = datareader_csv.load_feature_durations(durations_path)
    del feature_times['time_ms']
    del feature_times['class_label']
    plot_feature_barcharts(feature_times)
