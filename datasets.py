"""
Functions for creating the different datasets that can be used to train machine learning models.
Feature calculation is also carried out in this module.
"""
import datapoint as dp
import datareader_csv
import os
import csv
import math
import concurrent.futures as conf
from collections import deque
import datawriter_csv
import features


def __messages_to_datapoints(messages, window_ms, class_label, stride_ms, name=""):
    # Converts a list of messages to a list of DataPoints
    # where each point is comprised of 'messages' in 'window_ms' time window
    # as well as a dict of feature calculation durations.
    # 'stride_ms' determines how many milliseconds are to be elapsed between creation of two DataPoints.
    # That is, if 'stride_ms' is 50 and 'window_ms' is 100,
    # DataPoint 1 and 2 will share messages in half of their time windows.
    # 'class_label' determines whether intrusion was conducted in 'messages'
    if len(messages) == 0:
        return []

    windows = __find_windows(messages, window_ms, stride_ms)

    return features.windows_to_datapoints(windows, class_label, name)


def __find_windows(messages, window_ms, stride_ms):
    # Separates a list of messages into a list of windows.
    # Here, a window is defined as a list of messages within a specified timespan
    working_set = deque()

    working_set.append(messages[0])
    lowest_index = 0
    length = len(messages)

    # construct the initial working set. That is, the deque of messages used to create the next DataPoint.
    while lowest_index < length and \
            (messages[lowest_index].timestamp * 1000.0 - working_set[0].timestamp * 1000.0) <= window_ms:

        lowest_index += 1
        working_set.append(messages[lowest_index])

    lowest_index += 1
    old_time = working_set[0].timestamp

    windows = []
    for i in range(lowest_index, length):
        working_set.append(messages[i])
        time_expended = (working_set[len(working_set) - 1].timestamp - old_time) * 1000.0

        # repeatedly right-append to the working set,
        # until the time differences between the last message used for the previous DataPoint,
        # and the most recently appended message are offset by at least 'stride_ms' milliseconds.
        if time_expended >= stride_ms:
            low = working_set.popleft()

            # until the left-most and right-most messages in the working set are offset by at most 'window_ms',
            # left-pop a message from the working set.
            while (messages[i].timestamp * 1000.0 - low.timestamp * 1000.0) > window_ms:
                low = working_set.popleft()

            working_set.appendleft(low)
            windows.append(list(working_set))
            old_time = working_set[len(working_set) - 1].timestamp

    return windows


def __write_datapoints_csv(datapoints, window_ms, stride_ms, impersonation_split, dos_type, set_type):
    # Writes a list of DataPoints to file.
    # The file name and directory depends on the parameters.
    csv_path, directory = __get_dataset_path(window_ms, stride_ms, impersonation_split, dos_type, set_type)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(csv_path, "w", newline="") as datafile:
        datafile_writer = csv.writer(datafile, delimiter=",")

        # Writing the header.
        datafile_writer.writerow(dp.datapoint_attributes)

        for datapoint in datapoints:
            datafile_writer.writerow(dp.get_csv_row(datapoint))


def __concat_messages(msgs1, msgs2):
    # Joins two lists of messages,
    # by offsetting the timestamps of messages in the second list.
    offset = msgs1[len(msgs1) - 1].timestamp - msgs2[0].timestamp

    for msg in msgs2:
        msg.timestamp += offset

    return msgs1 + msgs2


def __neutralize_offset(messages):
    # Modifies input list of messages,
    # such that the first message starts at time 0.
    # Returns the changed input list.
    offset = messages[0].timestamp

    for message in messages:
        message.timestamp -= offset

    return messages


def __percentage_subset(collection, init, end):
    # Returns a subset of the specified collection, determined by the 'init' and 'end' percentages
    length = len(collection)
    init_index = math.floor(length * (init / 100.0))
    end_index = math.floor(length * (end / 100.0))

    return collection[init_index:end_index]


def __time_subset(messages, index_begin, subset_time_ms):
    # Returns a subset of a collection of messages, that span a range of subset_time ms beginning from index_begin
    timestamp_begin = messages[index_begin].timestamp
    index_end = index_begin

    for i in range(index_begin, len(messages)):
        if messages[i].timestamp > timestamp_begin + subset_time_ms / 1000:
            break
        index_end = i

    return messages[index_begin:index_end], index_end


def get_mixed_test(window_ms=100, stride_ms=100, imp_split=True, dos_type='original', verbose=False, in_parallel=True):
    """Constructs a list of test set DataPoints based on parameters.
    If this function is to be used from another file,
    all code must be wrapped in an __name__ == '__main__' check if used on a Windows system.

    :param window_ms: the window size (int ms).
    :param stride_ms: the step size (int ms).
    :param imp_split: a flag indicating whether the impersonation set has split labels.
    :param dos_type: a string indicating the type of DoS set used ('modified', 'original').
    :param verbose: a flag indicating how often progress should be output to console.
    :param in_parallel: a flag indicating whether features should be calculated in parallel.
    :return: a list of test DataPoints, a dictionary of feature durations.
    """

    # Load messages and remove time offsets
    attack_free_messages1 = __neutralize_offset(datareader_csv.load_attack_free1(verbose=verbose))
    attack_free_messages2 = __neutralize_offset(datareader_csv.load_attack_free2(verbose=verbose))
    fuzzy_messages = __neutralize_offset(datareader_csv.load_fuzzy(verbose=verbose))
    imp_messages1 = __neutralize_offset(datareader_csv.load_impersonation_1(verbose=verbose))
    imp_messages2 = __neutralize_offset(datareader_csv.load_impersonation_2(verbose=verbose))
    imp_messages3 = __neutralize_offset(datareader_csv.load_impersonation_3(verbose=verbose))

    if dos_type == "original":
        dos_messages = __neutralize_offset(datareader_csv.load_dos(verbose=verbose))
    else:
        dos_messages = __neutralize_offset(datareader_csv.load_modified_dos(verbose=verbose))

    raw_test_msgs = [
        (__percentage_subset(attack_free_messages1, 85, 100), "normal", "attack_free_1"),
        (__percentage_subset(attack_free_messages2, 85, 100), "normal", "attack_free_2"),
        (__percentage_subset(dos_messages, 85, 100), "dos", "dos"),
        (__percentage_subset(fuzzy_messages, 85, 100), "fuzzy", "fuzzy")]

    if imp_split:
        raw_test_msgs += [
            (__percentage_subset(imp_messages1[0:524052], 85, 100), "normal", "impersonation_normal_1"),
            (__percentage_subset(imp_messages1[524052:], 85, 100), "impersonation", "impersonation_attack_1"),
            (__percentage_subset(imp_messages2[0:484233], 85, 100), "normal", "impersonation_normal_2"),
            (__percentage_subset(imp_messages2[484233:], 85, 100), "impersonation", "impersonation_attack_2"),
            (__percentage_subset(imp_messages3[0:489677], 85, 100), "normal", "impersonation_normal_3"),
            (__percentage_subset(imp_messages3[489677:], 85, 100), "impersonation", "impersonation_attack_3")]
    else:
        raw_test_msgs += [
            (__percentage_subset(imp_messages1, 85, 100), "impersonation", "impersonation_1"),
            (__percentage_subset(imp_messages2, 85, 100), "impersonation", "impersonation_2"),
            (__percentage_subset(imp_messages3, 85, 100), "impersonation", "impersonation_3")]

    test_sets, feature_durations_list = __calculate_datapoints_from_sets(
        raw_test_msgs,
        window_ms, stride_ms,
        in_parallel)

    test_points = __collapse_datasets(test_sets)
    feature_durations = __get_feature_durations(feature_durations_list, test_points)

    return test_points, feature_durations


def get_mixed_training_validation(window_ms=100, stride_ms=100, imp_split=True,
                                  dos_type='original', verbose=False, in_parallel=True):
    """Constructs a training and validation set of DataPoints based on parameters.
    If this function is to be used from another file,
    all code must be wrapped in an __name__ == '__main__' check if used on a Windows system.

    :param window_ms: the window size (int ms).
    :param stride_ms: the step size (int ms).
    :param imp_split: a flag indicating whether the impersonation set has split labels.
    :param dos_type: a string indicating the type of DoS set used ('modified', 'original').
    :param verbose: a flag indicating how often progress should be output to console.
    :param in_parallel: a flag indicating whether features should be calculated in parallel.
    :return: a list of training DataPoints,, a list of validation DataPoints a dictionary of feature durations.
    """

    # Load messages and remove time offsets
    attack_free_messages1 = __neutralize_offset(datareader_csv.load_attack_free1(verbose=verbose))
    attack_free_messages2 = __neutralize_offset(datareader_csv.load_attack_free2(verbose=verbose))
    fuzzy_messages = __neutralize_offset(datareader_csv.load_fuzzy(verbose=verbose))
    imp_messages1 = __neutralize_offset(datareader_csv.load_impersonation_1(verbose=verbose))
    imp_messages2 = __neutralize_offset(datareader_csv.load_impersonation_2(verbose=verbose))
    imp_messages3 = __neutralize_offset(datareader_csv.load_impersonation_3(verbose=verbose))

    if dos_type == "original":
        dos_messages = __neutralize_offset(datareader_csv.load_dos(verbose=verbose))
    else:
        dos_messages = __neutralize_offset(datareader_csv.load_modified_dos(verbose=verbose))

    # Label raw datasets
    raw_training_msgs = [
        (__percentage_subset(attack_free_messages1, 0, 70), "normal", "attack_free_1"),
        (__percentage_subset(attack_free_messages2, 0, 70), "normal", "attack_free_2"),
        (__percentage_subset(dos_messages, 0, 70), "dos", "dos"),
        (__percentage_subset(fuzzy_messages, 0, 70), "fuzzy", "fuzzy")]

    raw_validation_msgs = [
        (__percentage_subset(attack_free_messages1, 70, 85), "normal", "attack_free_1"),
        (__percentage_subset(attack_free_messages2, 70, 85), "normal", "attack_free_2"),
        (__percentage_subset(dos_messages, 70, 85), "dos", "dos"),
        (__percentage_subset(fuzzy_messages, 70, 85), "fuzzy", "fuzzy")]

    if imp_split:
        raw_training_msgs += [
            (__percentage_subset(imp_messages1[0:517000], 0, 70), "normal", "impersonation_normal_1"),
            (__percentage_subset(imp_messages1[517000:], 0, 70), "impersonation", "impersonation_attack_1"),
            (__percentage_subset(imp_messages2[0:330000], 0, 70), "normal", "impersonation_normal_2"),
            (__percentage_subset(imp_messages2[330000:], 0, 70), "impersonation", "impersonation_attack_2"),
            (__percentage_subset(imp_messages3[0:534000], 0, 70), "normal", "impersonation_normal_3"),
            (__percentage_subset(imp_messages3[534000:], 0, 70), "impersonation", "impersonation_attack_3")]

        raw_validation_msgs += [
            (__percentage_subset(imp_messages1[0:517000], 70, 85), "normal", "impersonation_normal_1"),
            (__percentage_subset(imp_messages1[517000:], 70, 85), "impersonation", "impersonation_attack_1"),
            (__percentage_subset(imp_messages2[0:330000], 70, 85), "normal", "impersonation_normal_2"),
            (__percentage_subset(imp_messages2[330000:], 70, 85), "impersonation", "impersonation_attack_2"),
            (__percentage_subset(imp_messages3[0:534000], 70, 85), "normal", "impersonation_normal_3"),
            (__percentage_subset(imp_messages3[534000:], 70, 85), "impersonation", "impersonation_attack_3")]
    else:
        raw_training_msgs += [
            (__percentage_subset(imp_messages1, 0, 70), "impersonation", "impersonation_1"),
            (__percentage_subset(imp_messages2, 0, 70), "impersonation", "impersonation_2"),
            (__percentage_subset(imp_messages3, 0, 70), "impersonation", "impersonation_3")]

        raw_validation_msgs += [
            (__percentage_subset(imp_messages1, 70, 85), "impersonation", "impersonation_1"),
            (__percentage_subset(imp_messages2, 70, 85), "impersonation", "impersonation_2"),
            (__percentage_subset(imp_messages3, 70, 85), "impersonation", "impersonation_3")]

    training_sets, _ = __calculate_datapoints_from_sets(raw_training_msgs, window_ms, stride_ms, in_parallel)
    validation_sets, feature_durations_list = __calculate_datapoints_from_sets(
        raw_validation_msgs,
        window_ms, stride_ms,
        in_parallel)

    training_points = __collapse_datasets(training_sets)
    validation_points = __collapse_datasets(validation_sets)
    feature_durations = __get_feature_durations(feature_durations_list, validation_points)

    return training_points, validation_points, feature_durations


def __get_feature_durations(feature_durations_list, points):
    # Returns a dictionary of average times passed during calculation of features
    feature_durations = {}

    # Collapse resulting feature duration dicts into a single duration dict
    for attr in dp.datapoint_attributes:
        feature_durations[attr] = 0
        for durations in feature_durations_list:
            feature_durations[attr] += durations[attr]

        # Average feature duration
        feature_durations[attr] /= len(points)

    return feature_durations


def __collapse_datasets(datasets):
    # Returns the concatenated equivalent of the lists in 'datasets',
    # offsetting timestamps accordingly
    offset = 0
    points = []

    # collapse resulting lists of DataPoints into a single list of continuous timestamps
    for dataset in datasets:
        time_low = dataset[0].time_ms
        points += [__offset_datapoint(point, offset - time_low) for point in dataset]
        offset = points[len(points) - 1].time_ms

    return points


def __calculate_datapoints_from_sets(raw_msgs, window_ms, stride_ms, in_parallel=True):
    # Returns a list of DataPoint lists, calculated from the list of lists of Messages
    datasets = []
    feature_durations = []

    if in_parallel:
        # Start a new process for each list in 'raw_msgs' and calculate features and durations
        with conf.ProcessPoolExecutor() as executor:
            futures = {executor.submit(
                __messages_to_datapoints,
                tup[0],
                window_ms,
                tup[1],
                stride_ms,
                tup[2]) for tup in raw_msgs}

            for future in conf.as_completed(futures):
                datasets.append(future.result()[0])
                feature_durations.append(future.result()[1])
    else:
        # Calculate features for each list in 'raw_msgs' in sequence
        for tup in raw_msgs:
            dataset, times = __messages_to_datapoints(tup[0], window_ms, tup[1], stride_ms, tup[2])
            datasets.append(dataset)
            feature_durations.append(times)

    return datasets, feature_durations


def __offset_datapoint(point, offset):
    # Increments the timestamp of input DataPoint by the input offset and returns the DataPoint
    point.time_ms += offset

    return point


def __get_dataset_path(window_ms, stride_ms, impersonation_split, dos_type, set_type):
    # Returns the file and directory paths associated with input argument combination
    imp_name = "imp_split" if impersonation_split else "imp_full"
    name = f"mixed_{set_type}_{window_ms}ms_{stride_ms}ms"
    directory = f"data/feature/{imp_name}/{dos_type}/"

    return directory + name + ".csv", directory


def get_transitioning_dataset(window_ms=100, stride_ms=100, slice_sizes=[250, 250], verbose=False):
    """
    Return a list of datapoints, as well as the timestamps of transitions between attack free and impersonation
    :param window_ms: Window size to use
    :param stride_ms: Stride size to use
    :param slice_sizes: A list of sizes (in ms) for attack free and impersonation data. First size must be attack free
    :param verbose: Verbose parameter to pass to datareader_csv
    :return: datapoints list, transitions list
    """

    # Load datasets
    attack_free_messages = __neutralize_offset(datareader_csv.load_attack_free1(verbose=verbose))
    imp_messages = __neutralize_offset(datareader_csv.load_impersonation_1(verbose=verbose))[524052:]

    final_messages = None
    final_transitions = []
    attack_free_messages = attack_free_messages[math.floor(len(attack_free_messages) * 0.96):-1]
    imp_messages = imp_messages[math.floor(len(imp_messages) * 0.89):-1]
    attack_free_index = 0
    imp_index = 0
    transition_index = 0
    is_attack_free = True  # Bool to keep track of whether to current slice corresponds to a attack free or imp

    attack_free_messages = __neutralize_offset(attack_free_messages)
    imp_messages = __neutralize_offset(imp_messages)

    for size in slice_sizes:
        if is_attack_free:
            messages, attack_free_index = __time_subset(attack_free_messages, attack_free_index, size)
        else:
            messages, imp_index = __time_subset(imp_messages, imp_index, size)

        # Append to final_messages
        final_messages = messages if final_messages is None else __concat_messages(final_messages, messages)

        # Find transition timestamp
        transition_index = transition_index + len(messages) - 1
        transition_timestamp = final_messages[transition_index].timestamp * 1000  # Convert to ms
        final_transitions.append(transition_timestamp)

        is_attack_free = not is_attack_free

    # Calculate datapoints from found messages
    datapoints, _ = __messages_to_datapoints(final_messages, window_ms, 'normal', stride_ms)

    current_slice_index = 0
    current_upper = 0
    for datapoint in datapoints:
        if datapoint.time_ms > current_upper:
            current_upper += slice_sizes[current_slice_index]
            current_slice_index += 1
        datapoint.class_label = 'impersonation' if current_slice_index % 2 == 0 else 'normal'

    return datapoints, final_transitions[0:-1]


def load_or_create_datasets(window_ms=100, stride_ms=100, imp_split=True, dos_type='original',
                            force_create=False, verbose=False, in_parallel=True):
    """Returns the training and validation sets associated with input argument combination.
    If the datasets do not exist, they are created and saved in the process.

    :param window_ms: the window size (int ms).
    :param stride_ms: the step size (int ms).
    :param imp_split: a flag indicating whether the impersonation dataset has split labels.
    :param dos_type: a string indicating the DoS dataset used ('modified', 'original').
    :param force_create: a flag indicating whether the dataset should be generated, even if it exists.
    :param verbose: a flag indicating how much progress should be output to console.
    :param in_parallel: a flag indicating whether features should be calculated in parallel.
    :return: a list of training DataPoints, a list of validation DataPoints, a dictionary of feature durations.
    """

    training_name, _ = __get_dataset_path(window_ms, stride_ms, imp_split, dos_type, 'training')
    validation_name, _ = __get_dataset_path(window_ms, stride_ms, imp_split, dos_type, 'validation')
    time_path, directory = __get_dataset_path(window_ms, stride_ms, imp_split, dos_type, 'validation_time')

    # Load the datasets if they exist.
    if os.path.exists(training_name) and os.path.exists(validation_name) and not force_create:
        training_set = datareader_csv.load_datapoints(training_name, verbose=verbose)
        validation_set = datareader_csv.load_datapoints(validation_name, verbose=verbose)
        feature_durations = datareader_csv.load_feature_durations(time_path)
    else:
        # Create and save the datasets otherwise.
        training_set, validation_set, feature_durations = get_mixed_training_validation(
            window_ms, stride_ms,
            imp_split, dos_type,
            verbose=verbose,
            in_parallel=in_parallel)

        __write_datapoints_csv(training_set, window_ms, stride_ms, imp_split, dos_type, 'training')
        __write_datapoints_csv(validation_set, window_ms, stride_ms, imp_split, dos_type, 'validation')
        datawriter_csv.save_feature_durations(feature_durations, time_path, directory)

    return training_set, validation_set, feature_durations
