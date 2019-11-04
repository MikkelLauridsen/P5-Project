import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import datasets as ds
import datapoint


def class_to_color(cls):
    return {
        "normal": "#246EB6",
        "dos": "#B62121",
        "fuzzy": "#0A813E",
        "impersonation": "#FF7A0E"
    }.get(cls, "#000000")


# Takes all the features defined in DataPoint and plots them
def plot_all_features(datapoints):
    time_ms = [point.time_ms for point in datapoints]

    # Extract features from DataPoints
    is_injected = [point.is_injected for point in datapoints]

    for attr in datapoint.datapoint_attributes:
        feature_list = [getattr(point, attr) for point in datapoints]
        # Get attribute description if available
        feature_description = datapoint.datapoint_attribute_descriptions.get(attr, attr)

        if attr == "time_ms" or attr == "is_injected":
            pass

        # Special case for mean_id_intervals_variance and req_to_res_time_variance, to manually set axis limits
        elif attr == "mean_id_intervals_variance" or attr == "req_to_res_time_variance":
            setup_scatter(time_ms, feature_list, "Time", feature_description, is_injected, False)
            plt.ylim(top=0.0005, bottom=-0.00025)  # Manually set axis limits
            plt.show()
        else:
            setup_scatter(time_ms, feature_list, "Time", feature_description, is_injected)


# Helper function for setting up a scatter plot.
# The function does not call plt.show() in order to allow further configuration afterwards.
def setup_scatter(xaxis, yaxis, xlabel, ylabel, is_injected, show=True):
    colors = [class_to_color(cls) for cls in is_injected]
    legends = [(class_to_color("normal"), "attack free state"),
               (class_to_color("dos"), "DoS attack state"),
               (class_to_color("fuzzy"), "Fuzzy attack state"),
               (class_to_color("impersonation"), "Impersonation attack state")]

    patches = []

    for legend in legends:
        patches += [mpatches.Patch(color=legend[0], label=legend[1])]

    plt.figure(figsize=(12, 7))
    plt.scatter(xaxis, yaxis, s=5, c=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles=patches)

    if show:
        plt.show()


if __name__ == "__main__":
    training_points, test_points, _ = ds.load_or_create_datasets(period_ms=100,
                                                                 stride_ms=100,
                                                                 impersonation_split=False,
                                                                 dos_type='modified')

    plot_all_features(training_points)
