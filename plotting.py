import datareader_csv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import id_based_datasets
import idpoint

#training_set, validation_set, test_set = id_based_datasets.get_mixed_datasets(100)
#idpoints = training_set

idpoints = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_32888_100ms.csv", 0)
#idpoints = datareader_csv.load_idpoints("idpoint_dataset/mixed_validation_7046_100ms.csv", 0)
#idpoints = datareader_csv.load_idpoints("idpoint_dataset/mixed_test_7053_100ms.csv", 0)

#dataset = datareader_csv.load_impersonation_3()
#idpoints = id_based_datasets.messages_to_idpoints(dataset[0:533000], 100, False)
#idpoints += id_based_datasets.messages_to_idpoints(dataset[533000:], 100, True)

#dataset = datareader_csv.load_fuzzy()
#idpoints = id_based_datasets.messages_to_idpoints(dataset[0:450000], 100, False)
#idpoints += id_based_datasets.messages_to_idpoints(dataset[450000:], 100, True)

#dataset = datareader_csv.load_dos()
#idpoints = id_based_datasets.messages_to_idpoints(dataset, 100, True)

#dataset = datareader_csv.load_attack_free1()
#idpoints = id_based_datasets.messages_to_idpoints(dataset, 100, False)


def class_to_color(cls):
    if cls == "normal":
        return "#246EB6"
    elif cls == "dos":
        return "#B62121"
    elif cls == "fuzzy":
        return "#0A813E"
    elif cls == "impersonation":
        return "#FF7A0E"
    else:
        raise ValueError()


# Extract features from idpoints
is_injected = [idpoint.is_injected for idpoint in idpoints]

scatter_default_colors = [class_to_color(cls) for cls in is_injected]
scatter_default_legends = [(class_to_color("normal"), "attack free state"),
                           (class_to_color("dos"), "DoS attack state"),
                           (class_to_color("fuzzy"), "Fuzzy attack state"),
                           (class_to_color("impersonation"), "Impersonation attack state")]


# Helper function for setting up a scatter plot.
# The function does not call plt.show() in order to allow further configuration afterwards.
def setup_scatter(xaxis, yaxis, xlabel, ylabel, show=True, colors=scatter_default_colors, legends=scatter_default_legends):
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


# Takes all the features defined in IDPoint and plots them
def plot_all_features(idpoints):
    time_ms = [point.time_ms for point in idpoints]

    for attr in idpoint.idpoint_attributes:
        feature_list = [getattr(idp, attr) for idp in idpoints]
        feature_description = idpoint.idpoint_attribute_descriptions.get(attr, attr)  # Get attribute desciption if available

        if attr == "time_ms" or attr == "is_injected":
            pass

        # Special case for mean_id_intervals_variance and req_to_res_time_variance, to manually set axis limits
        elif attr == "mean_id_intervals_variance" or attr == "req_to_res_time_variance":
            setup_scatter(time_ms, feature_list, "Time", feature_description, False)
            plt.ylim(top=0.0005, bottom=-0.00025)  # Manually set axis limits
            plt.show()
        else:
            setup_scatter(time_ms, feature_list, "Time", feature_description)


plot_all_features(idpoints)
