import datareader_csv
import matplotlib.pyplot as plt
import id_based_datasets

#training_set, validation_set, test_set = id_based_datasets.get_mixed_datasets(100)
#idpoints = training_set

idpoints = datareader_csv.load_idpoints("idpoint_dataset/mixed_training_32888_100ms.csv", 0)
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


time_ms = [idpoint.time_ms for idpoint in idpoints]
is_injected = [idpoint.is_injected for idpoint in idpoints]
mean_id_interval = [idpoint.mean_id_interval for idpoint in idpoints]
variance_id_frequency = [idpoint.variance_id_frequency for idpoint in idpoints]
num_id_transitions = [idpoint.num_id_transitions for idpoint in idpoints]
num_ids = [idpoint.num_ids for idpoint in idpoints]
num_msgs = [idpoint.num_msgs for idpoint in idpoints]
mean_id_intervals_variances = [idpoint.mean_id_intervals_variance for idpoint in idpoints]
mean_data_bit_counts = [idpoint.mean_data_bit_count for idpoint in idpoints]
variance_data_bit_counts = [idpoint.variance_data_bit_count for idpoint in idpoints]
mean_variance_data_bit_count_ids = [idpoint.mean_variance_data_bit_count_id for idpoint in idpoints]

injected_colors = ["#B62121" if inj else "#246EB6" for inj in is_injected]


def setup_scatter(xaxis, yaxis, xlabel, ylabel, colors=injected_colors, legend=""):
    plt.figure(figsize=(12, 7))
    plt.scatter(xaxis, yaxis, s=5, c=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)


setup_scatter(time_ms, mean_id_interval, "Time", "Mean id interval")
plt.show()

setup_scatter(time_ms, variance_id_frequency, "Time", "Variance id frequency")
plt.show()

setup_scatter(time_ms, num_id_transitions, "Time", "# id transitions")
plt.show()

setup_scatter(time_ms, num_ids, "Time", "# ids")
plt.show()

setup_scatter(time_ms, num_msgs, "Time", "# messages")
plt.show()

setup_scatter(time_ms, mean_id_intervals_variances, "Time", "mean_id_intervals_variances")
plt.ylim(top=0.0005, bottom=-0.00025)
plt.show()

setup_scatter(time_ms, mean_data_bit_counts, "Time", "Mean data bit-counts")
plt.show()

setup_scatter(time_ms, variance_data_bit_counts, "Time", "Variance data bit-counts")
plt.show()

setup_scatter(time_ms, mean_variance_data_bit_count_ids, "Time", "Mean variance data bit-count ids")
plt.show()

plt.figure(figsize=(10, 5))
plt.stackplot(time_ms, is_injected)
plt.xlabel("Time")
plt.ylabel("Is injected")
plt.show()
