"""
Functions for creating the different datasets that can be used to train machine learning models.
Feature calculation is also carried out in this module.
"""
import datapoint as dp
import datareader_csv
import os
import csv
import math
import time
import concurrent.futures as conf
from collections import deque
import datawriter_csv
import features


# Converts a list of messages to a list of DataPoints
# where each point is comprised of 'messages' in 'period_ms' time window
# as well as a dict of feature calculation durations.
# 'stride_ms' determines how many milliseconds are to be elapsed between creation of two DataPoints.
# That is, if 'stride_ms' is 50 and 'period_ms' is 100,
# DataPoint 1 and 2 will share messages in half of their time windows.
# 'class_label' determines whether intrusion was conducted in 'messages'
def messages_to_datapoints(messages, period_ms, class_label, stride_ms, name=""):
    if len(messages) == 0:
        return []

    windows = __find_windows(messages, period_ms, stride_ms)
    return features.windows_to_datapoints(windows, class_label, name)


# Separates a list of messages into a list of windows.
# A window is here considered a list of messages within a specified timespan
def __find_windows(messages, period_ms, stride_ms):
    working_set = deque()

    working_set.append(messages[0])
    lowest_index = 0
    length = len(messages)

    # construct the initial working set. That is, the deque of messages used to create the next DataPoint.
    while lowest_index < length and \
            (messages[lowest_index].timestamp * 1000.0 - working_set[0].timestamp * 1000.0) <= period_ms:

        lowest_index += 1
        working_set.append(messages[lowest_index])

    lowest_index += 1
    old_time = working_set[0].timestamp

    windows = []
    for i in range(lowest_index, length):
        working_set.append(messages[i])
        time_expended = (working_set[len(working_set) - 1].timestamp - old_time) * 1000.0

        # repeatedly right-append to the working set,
        # until the time period between the last message used for the previous DataPoint,
        # and the most recently appended message are offset by at least 'stride_ms' milliseconds.
        if time_expended >= stride_ms:
            low = working_set.popleft()

            # until the left-most and right-most messages in the working set are offset by at most 'period_ms',
            # left-pop a message from the working set.
            while (messages[i].timestamp * 1000.0 - low.timestamp * 1000.0) > period_ms:
                low = working_set.popleft()

            working_set.appendleft(low)
            windows.append(list(working_set))
            old_time = working_set[len(working_set) - 1].timestamp

    return windows


# Writes a list of DataPoints to file.
# The file name and directory depends on the parameters.
def write_datapoints_csv(datapoints, period_ms, stride_ms, impersonation_split, dos_type, set_type):
    csv_path, dir = get_dataset_path(period_ms, stride_ms, impersonation_split, dos_type, set_type)

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(csv_path, "w", newline="") as datafile:
        datafile_writer = csv.writer(datafile, delimiter=",")

        # Writing the header.
        datafile_writer.writerow(dp.datapoint_attributes)

        for datapoint in datapoints:
            datafile_writer.writerow(dp.get_csv_row(datapoint))


# Joins two lists of messages,
# by offsetting the timestamps of messages in the second list.
def concat_messages(msgs1, msgs2):
    offset = msgs1[len(msgs1) - 1].timestamp

    for msg in msgs2:
        msg.timestamp += offset

    return msgs1 + msgs2


# Modifies input list of messages,
# such that the first message starts at time 0.
# returns the changed input list.
def neutralize_offset(messages):
    offset = messages[0].timestamp

    for message in messages:
        message.timestamp -= offset

    return messages


def __percentage_subset(collection, init, end):
    length = len(collection)
    init_index = math.floor(length * (init / 100.0))
    end_index = math.floor(length * (end / 100.0))


    return collection[init_index:end_index]


def get_mixed_test(period_ms=100, stride_ms=100, impersonation_split=True, dos_type='original', verbose=False):
    """Constructs a list of test set DataPoints based on parameters.
    :returns
        a list of DataPoints.
        a dictionary of feature durations.

    :parameter
        'period_ms' determines the duration of the time window used to create each DataPoint.
        'stride_ms' determines how little of the previous time window may be used to create the next DataPoint.
        'impersonation_split' dictates whether the raw impersonation datasets,
            should be separated in attack free and attack affected data.
        'dos_type' determines which DoS dataset should be used: 'original', 'modified'.

    Constructed from 15% of each raw dataset.
    If this function is to be used from another file,
    all code must be wrapped in an __name__ == '__main__' check if used on a Windows system.
    """

    # load messages and remove time offsets
    attack_free_messages1 = neutralize_offset(datareader_csv.load_attack_free1(verbose=verbose))
    attack_free_messages2 = neutralize_offset(datareader_csv.load_attack_free2(verbose=verbose))
    fuzzy_messages = neutralize_offset(datareader_csv.load_fuzzy(verbose=verbose))
    imp_messages1 = neutralize_offset(datareader_csv.load_impersonation_1(verbose=verbose))
    imp_messages2 = neutralize_offset(datareader_csv.load_impersonation_2(verbose=verbose))
    imp_messages3 = neutralize_offset(datareader_csv.load_impersonation_3(verbose=verbose))
    dos_messages = neutralize_offset(datareader_csv.load_dos(verbose=verbose) if dos_type == 'original' else
                                     datareader_csv.load_modified_dos(verbose=verbose))

    raw_test_msgs = [
        (__percentage_subset(attack_free_messages1, 85, 100), "normal", "attack_free_1"),
        (__percentage_subset(attack_free_messages2, 85, 100), "normal", "attack_free_2"),
        (__percentage_subset(dos_messages, 85, 100), "dos", "dos"),
        (__percentage_subset(fuzzy_messages, 85, 100), "fuzzy", "fuzzy")]

    if impersonation_split:
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

    test_sets, feature_durations_list = calculate_datapoints_from_sets(raw_test_msgs, period_ms, stride_ms)
    test_points = collapse_datasets(test_sets)
    feature_durations = get_feature_durations(feature_durations_list, test_points)

    return test_points, feature_durations


def get_mixed_training_validation(period_ms=100, stride_ms=100, impersonation_split=True,
                                  dos_type='original', verbose=False, in_parallel=True):
    """Constructs a training and validation set of DataPoints based on parameters.
        :returns
            a list of DataPoints corresponding to the training data.
            a list of DataPoints corresponding to the validation data.
            a dictionary of feature durations.

        :parameter
            'period_ms' determines the duration of the time window used to create each DataPoint.
            'stride_ms' determines how little of the previous time window may be used to create the next DataPoint.
            'impersonation_split' dictates whether the raw impersonation datasets,
                should be separated in attack free and attack affected data.
            'dos_type' determines which DoS dataset should be used: 'original', 'modified'.

        Constructed from 15% of each raw dataset.
        If this function is to be used from another file,
        all code must be wrapped in an __name__ == '__main__' check if used on a Windows system.
    """

    # load messages and remove time offsets
    attack_free_messages1 = neutralize_offset(datareader_csv.load_attack_free1(verbose=verbose))
    attack_free_messages2 = neutralize_offset(datareader_csv.load_attack_free2(verbose=verbose))
    fuzzy_messages = neutralize_offset(datareader_csv.load_fuzzy(verbose=verbose))
    imp_messages1 = neutralize_offset(datareader_csv.load_impersonation_1(verbose=verbose))
    imp_messages2 = neutralize_offset(datareader_csv.load_impersonation_2(verbose=verbose))
    imp_messages3 = neutralize_offset(datareader_csv.load_impersonation_3(verbose=verbose))
    dos_messages = neutralize_offset(datareader_csv.load_dos(verbose=verbose) if dos_type == 'original' else
                                     datareader_csv.load_modified_dos(verbose=verbose))

    # label raw datasets
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

    if impersonation_split:
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

    training_sets, _ = calculate_datapoints_from_sets(raw_training_msgs, period_ms, stride_ms, in_parallel)
    validation_sets, feature_durations_list = calculate_datapoints_from_sets(
        raw_validation_msgs,
        period_ms, stride_ms,
        in_parallel)

    training_points = collapse_datasets(training_sets)
    validation_points = collapse_datasets(validation_sets)
    feature_durations = get_feature_durations(feature_durations_list, validation_points)

    return training_points, validation_points, feature_durations


def get_feature_durations(feature_durations_list, points):
    feature_durations = {}

    # Collapse resulting feature duration dicts into a single duration dict
    for attr in dp.datapoint_attributes:
        feature_durations[attr] = 0
        for durations in feature_durations_list:
            feature_durations[attr] += durations[attr]

        # Average feature duration
        feature_durations[attr] /= len(points)

    return feature_durations


def collapse_datasets(datasets):
    offset = 0
    points = []

    # collapse resulting lists of DataPoints into a single list of continuous timestamps
    for dataset in datasets:
        time_low = dataset[0].time_ms
        points += [offset_datapoint(point, offset - time_low) for point in dataset]
        offset = points[len(points) - 1].time_ms

    return points


def calculate_datapoints_from_sets(raw_msgs, period_ms, stride_ms, in_parallel=True):
    datasets = []
    feature_durations = []

    if in_parallel:
        with conf.ProcessPoolExecutor() as executor:
            futures = {executor.submit(
                messages_to_datapoints,
                tup[0],
                period_ms,
                tup[1],
                stride_ms,
                tup[2]) for tup in raw_msgs}

            for future in conf.as_completed(futures):
                datasets.append(future.result()[0])
                feature_durations.append(future.result()[1])
    else:
        for tup in raw_msgs:
            dataset, times = messages_to_datapoints(tup[0], period_ms, tup[1], stride_ms, tup[2])
            datasets.append(dataset)
            feature_durations.append(times)

    return datasets, feature_durations


# Increments the timestamp of input DataPoint by the input offset and returns the DataPoint
def offset_datapoint(point, offset):
    point.time_ms += offset

    return point


# Returns the file and directory paths associated with input argument combination.
def get_dataset_path(period_ms, stride_ms, impersonation_split, dos_type, set_type):
    imp_name = "imp_split" if impersonation_split else "imp_full"
    name = f"mixed_{set_type}_{period_ms}ms_{stride_ms}ms"
    dir = f"data/feature/{imp_name}/{dos_type}/"

    return dir + name + ".csv", dir


# Returns the training and validation sets associated with input argument combination.
# If the datasets do not exist, they are created and saved in the process.
def load_or_create_datasets(period_ms=100, stride_ms=100, imp_split=True, dos_type='original',
                            force_create=False, verbose=False, in_parallel=True):

    training_name, _ = get_dataset_path(period_ms, stride_ms, imp_split, dos_type, 'training')
    validation_name, _ = get_dataset_path(period_ms, stride_ms, imp_split, dos_type, 'validation')
    time_path, dir = get_dataset_path(period_ms, stride_ms, imp_split, dos_type, 'validation_time')

    # load the datasets if they exist.
    if os.path.exists(training_name) and os.path.exists(validation_name) and not force_create:
        training_set = datareader_csv.load_datapoints(training_name, verbose=verbose)
        validation_set = datareader_csv.load_datapoints(validation_name, verbose=verbose)
        feature_durations = datareader_csv.load_feature_durations(time_path)
    else:
        # create and save the datasets otherwise.
        training_set, validation_set, feature_durations = get_mixed_training_validation(
            period_ms, stride_ms,
            imp_split, dos_type,
            verbose=verbose,
            in_parallel=in_parallel)

        write_datapoints_csv(training_set, period_ms, stride_ms, imp_split, dos_type, 'training')
        write_datapoints_csv(validation_set, period_ms, stride_ms, imp_split, dos_type, 'validation')
        datawriter_csv.save_feature_durations(feature_durations, time_path, dir)

    return training_set, validation_set, feature_durations
