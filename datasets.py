import datapoint as dp
import datareader_csv
import os
import csv
import math
import time
import functools as ft
import concurrent.futures as conf
from collections import deque
from sklearn.model_selection import train_test_split


# Calculates the skewness of the values in input list.
# Pearson's second skewness coefficient is used as equation.
# If the list has less than two elements, the skewness will default to 0.
def __calculate_skewness(values):
    if len(values) == 0:
        return 0

    values.sort()  # sort values, such that the median can be found

    mean = math.fsum(values) / len(values)
    median = values[math.floor(len(values) / 2)]
    variance = __calculate_variance(values)

    if variance == 0:
        return 0

    return (3 * (mean - median)) / math.sqrt(variance)


# Calculates the kurtosis of the values in input list.
# Bock's kurtosis coefficient is used, which means the kurtosis of the normal distribution is 3.
# The kurtosis also defaults to 3 if input list has less than two elements.
def __calculate_kurtosis(values):
    if len(values) == 0:
        return 3

    n = len(values)
    avg = math.fsum(values) / n
    s2 = ft.reduce(lambda y, x: (x - avg) ** 2 + y, values, 0)  # find sum of squared deviations
    s4 = ft.reduce(lambda y, x: (x - avg) ** 4 + y, values, 0)  # fund sum of deviations raised to the fourth power
    m2 = s2 / n
    m4 = s4 / n

    if m2 == 0:
        return 3

    return m4 / (m2 ** 2)


# Calculates the variance of elements in input list.
# The equation for population variance is used.
# Defaults to 0 if the input list has less than 2 elements.
def __calculate_variance(values):
    avg_freq = math.fsum(values) / len(values)
    variance = 0

    for freq in values:
        deviation = freq - avg_freq
        variance += deviation * deviation

    return (1.0 / len(values)) * variance


# For each bit in the CAN-bus data-field,
# calculates the probability of the bit having value 1,
# based on data-fields in input list of messages.
# If the DLC has value 4, at most 32 bits can have value 1.
# Returns a list of the resulting probabilities.
def __calculate_probability_bits(messages):
    bits = [0.0 for i in range(64)]

    for message in messages:
        if message.dlc != 0:
            for i in range(len(message.data)):
                for j in range(8):
                    if message.data[i] & (0b10000000 >> j):
                        bits[i * 8 + j] += 1.0

    for i in range(64):
        bits[i] = bits[i] / len(messages)

    return bits


# Counts the number of 1s in the data-field of input message.
def __calculate_bit_count(message):
    if message.dlc == 0:
        return 0

    count = 0

    for byte in message.data:
        for i in range(8):
            if byte & (0b10000000 >> i):
                count += 1

    return count


# Returns a dictionary where keys are ids and values are lists of the intervals between messages of the same id
def __find_id_intervals(messages):
    id_timestamp_intervals = {}
    last_seen_timestamps = {}

    for message in messages:
        if message.id in last_seen_timestamps:
            interval = message.timestamp - last_seen_timestamps[message.id]
            id_timestamp_intervals.setdefault(message.id, [])
            id_timestamp_intervals[message.id].append(interval)

        last_seen_timestamps[message.id] = message.timestamp

    return id_timestamp_intervals


# For each unique ID in input list of messages,
# constructs a list of time periods between two messages with this ID.
# Afterwards, the variance each list is found.
# Finally, the skewness of these variances is calculated and returned.
def calculate_skewness_id_interval_variances(messages):
    id_timestamp_intervals = __find_id_intervals(messages)

    intervals_variances = []
    for intervals in id_timestamp_intervals.values():
        intervals_variances.append(__calculate_variance(intervals))

    return __calculate_skewness(intervals_variances)


# For each unique ID, constructs a list of bit-counts calculated from input list of messages.
# Afterwards, the variance of each list is found.
# Finally, the kurtosis of these variances is calculated and returned.
def calculate_kurtosis_variance_data_bit_count_id(messages):
    id_counts = {}

    for message in messages:
        if message.id in id_counts:
            id_counts[message.id].append(__calculate_bit_count(message))
        else:
            id_counts[message.id] = [__calculate_bit_count(message)]

    variances = []

    for counts in id_counts.values():
        variances.append(__calculate_variance(counts))

    return __calculate_kurtosis(variances)


# For each bit in the CAN-bus data-field, calculates the probability of it having value 1,
# based on input list of messages.
# Finally, calculates and returns the mean of these probabilities.
def calculate_mean_probability_bits(messages):
    bits = __calculate_probability_bits(messages)

    return math.fsum(bits) / 64.0


# For each unique ID in input list of messages,
# constructs a list of time periods between messages of this ID.
# Afterwards, calculates the variance of elements in each list, separately.
# Finally, calculates and returns the mean of these variances.
# Returned value defaults to 0 if no intervals were present in the time window.
def calculate_mean_id_intervals_variance(messages):
    id_timestamp_intervals = __find_id_intervals(messages)

    intervals_variances = []
    for intervals in id_timestamp_intervals.values():
        intervals_variances.append(__calculate_variance(intervals))

    return 0 if len(intervals_variances) == 0 else math.fsum(intervals_variances) / len(intervals_variances)


# Finds the bit-count of each message in input list, separately.
# Calculates and returns the mean bit-count.
def calculate_mean_data_bit_count(messages):
    counts = []

    for message in messages:
        counts.append(__calculate_bit_count(message))

    return math.fsum(counts) / len(counts)


# Finds the bit-count of each message in input list, separately.
# Calculates and returns the population variance of the bit-counts.
def calculate_variance_data_bit_count(messages):
    counts = []

    for message in messages:
        counts.append(__calculate_bit_count(message))

    return __calculate_variance(counts)


# For each unique ID in input list of messages,
# Constructs a list of bit-counts, for messages of this ID.
# Afterwards, calculates the variance of these bit-counts, separated by ID.
# Finally, calculates and returns the mean variance.
# Returned value defaults to 0 if no messages were received in the time window.
def calculate_mean_variance_data_bit_count_id(messages):
    id_counts = {}

    for message in messages:
        if message.id in id_counts:
            id_counts[message.id].append(__calculate_bit_count(message))
        else:
            id_counts[message.id] = [__calculate_bit_count(message)]

    variances = []

    for counts in id_counts.values():
        variances.append(__calculate_variance(counts))

    return 0 if len(variances) == 0 else math.fsum(variances) / len(variances)


# Finds and returns the mean ID interval,
# where an ID interval is the time period between two messages of the same ID.
def calculate_mean_id_interval(messages):
    intervals = []
    last_seen_timestamps = {}

    for message in messages:
        if message.id in last_seen_timestamps:
            intervals.append(message.timestamp - last_seen_timestamps[message.id])

        last_seen_timestamps[message.id] = message.timestamp

    return 0 if len(intervals) == 0 else math.fsum(intervals) / len(intervals)


# Finds and returns the variance of ID frequencies in 'messages',
# where a frequency is the number of times a given ID was used in 'messages'.
def calculate_variance_id_frequency(messages):
    frequencies = {}

    for message in messages:
        if message.id not in frequencies:
            frequencies[message.id] = 1
        else:
            frequencies[message.id] += 1

    values = frequencies.values()
    return __calculate_variance(values)


# Finds and returns the number of unique ID transitions in 'messages',
# where (msg1.ID -> msg2.ID) is a transition.
def calculate_num_id_transitions(messages):
    if len(messages) == 0:
        return 0

    transitions_seen = set()
    previous_id = messages[0].id

    for message in messages[1:]:
        transitions_seen.add((previous_id, message.id))
        previous_id = message.id

    return len(transitions_seen)


# Finds and returns the number of unique IDs in 'messages'
def calculate_num_ids(messages):
    ids_seen = set()

    for message in messages:
        ids_seen.add(message.id)

    return len(ids_seen)


# Constructs a list of time periods between remote frames and responses in input list of messages.
# Afterwards, the variance of these time periods is calculated and returned.
# The returned value defaults to 0, if no request frame was present or responded to in the time window.
def calculate_req_to_res_time_variance(messages):
    intervals = []
    latest_remote_frame_timestamp = {}

    for message in messages:
        if message.rtr == 0b100:
            latest_remote_frame_timestamp[message.id] = message.timestamp
        elif message.rtr == 0b000 and latest_remote_frame_timestamp.get(message.id, None) is not None:
            intervals.append(message.timestamp - latest_remote_frame_timestamp[message.id])
            latest_remote_frame_timestamp[message.id] = None

    return 0 if len(intervals) == 0 else __calculate_variance(intervals)


# Constructs a list of time periods between messages of the same ID.
# Afterwards, calculates and returns the kurtosis of these time periods.
def calculate_kurtosis_id_interval(messages):
    intervals = []
    last_seen_timestamps = {}

    for message in messages:
        if message.id in last_seen_timestamps:
            intervals.append(message.timestamp - last_seen_timestamps[message.id])

        last_seen_timestamps[message.id] = message.timestamp

    return __calculate_kurtosis(intervals)


# Calculates the frequency of each ID in input list of messages.
# Afterwards, calculates and returns the skewness of these frequencies.
def calculate_skewness_id_frequency(messages):
    frequencies = {}

    for message in messages:
        if message.id not in frequencies:
            frequencies[message.id] = 1
        else:
            frequencies[message.id] += 1

    values = list(frequencies.values())

    return __calculate_skewness(values)


# Calculates the frequency of each ID in input list of messages.
# Afterwards, calculates and returns the kurtosis of these frequencies.
def calculate_kurtosis_id_frequency(messages):
    frequencies = {}

    for message in messages:
        if message.id not in frequencies:
            frequencies[message.id] = 1
        else:
            frequencies[message.id] += 1

    values = frequencies.values()

    return __calculate_kurtosis(values)


# For each unique ID in input list of messages,
# constructs a list of time periods between messages with this ID.
# Afterwards, calculates the means of elements of these lists, separately.
# Finally, calculates and returns the kurtosis of these means.
def calculate_kurtosis_mean_id_intervals(messages):
    id_timestamp_intervals = __find_id_intervals(messages)

    interval_means = []
    for intervals in id_timestamp_intervals.values():
        interval_means.append(sum(intervals) / len(intervals))

    return __calculate_kurtosis(interval_means)


# Constructs a list of time periods between remote frames and responses in input list of messages.
# Afterwards, calculates and returns the kurtosis of these time periods.
def calculate_kurtosis_req_to_res_time(messages):
    intervals = []
    latest_remote_frame_timestamp = {}

    for message in messages:
        if message.rtr == 0b100:
            latest_remote_frame_timestamp[message.id] = message.timestamp
        elif message.rtr == 0b000 and latest_remote_frame_timestamp.get(message.id, None) is not None:
            intervals.append(message.timestamp - latest_remote_frame_timestamp[message.id])
            latest_remote_frame_timestamp[message.id] = None

    return __calculate_kurtosis(intervals)


# Converts a list of messages to a list of DataPoints
# where each point is comprised of 'messages' in 'period_ms' time window
# as well as a dict of feature calculation durations.
# 'stride_ms' determines how many milliseconds are to be elapsed between creation of two DataPoints.
# That is, if 'stride_ms' is 50 and 'period_ms' is 100,
# DataPoint 1 and 2 will share messages in half of their time windows.
# 'is_injected' determines whether intrusion was conducted in 'messages'
def messages_to_datapoints(messages, period_ms, is_injected, stride_ms, name=""):
    if len(messages) == 0:
        return []

    windows = __find_windows(messages, period_ms, stride_ms)
    return __windows_to_datapoints(windows, is_injected, name)


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


# Calculates a list of datapoints from a list of windows (a window being a list of messages)
# 'is_injected' determines whether intrusion was conducted in 'messages'
# this function may never be called with an empty list
def __windows_to_datapoints(windows, is_injected, name):
    # maps a function to an attribute. The function must accept a list of messages.
    # missing mappings are allowed, and will give the feature a value of 0
    attribute_function_mappings = {
        "time_ms": lambda msgs: msgs[0].timestamp * 1000,
        "is_injected": lambda msgs: is_injected,
        "mean_id_interval": calculate_mean_id_interval,
        "variance_id_frequency": calculate_variance_id_frequency,
        "num_id_transitions": calculate_num_id_transitions,
        "num_ids": calculate_num_ids,
        "num_msgs": len,
        "mean_id_intervals_variance": calculate_mean_id_intervals_variance,
        "mean_data_bit_count": calculate_mean_data_bit_count,
        "variance_data_bit_count": calculate_variance_data_bit_count,
        "mean_variance_data_bit_count_id": calculate_mean_variance_data_bit_count_id,
        "mean_probability_bits": calculate_mean_probability_bits,
        "req_to_res_time_variance": calculate_req_to_res_time_variance,
        "kurtosis_id_interval": calculate_kurtosis_id_interval,
        "kurtosis_id_frequency": calculate_kurtosis_id_frequency,
        "kurtosis_mean_id_intervals": calculate_kurtosis_mean_id_intervals,
        "kurtosis_variance_data_bit_count_id": calculate_kurtosis_variance_data_bit_count_id,
        "skewness_id_interval_variances": calculate_skewness_id_interval_variances,
        "skewness_id_frequency": calculate_skewness_id_frequency,
        "kurtosis_req_to_res_time": calculate_kurtosis_req_to_res_time
    }

    datapoints = []

    # Fill datapoint list with blank datapoints
    for i in range(len(windows)):
        datapoints.append(dp.DataPoint(*[0 for attr in dp.datapoint_attributes]))

    durations = {}

    # Populate datapoints by adding features one by one
    # Feature values are calculated on a per feature basis
    for i, attr in enumerate(dp.datapoint_attributes):
        print(f"{name} Calculating feature {attr} ({i + 1})")
        feature_func = attribute_function_mappings[attr]

        time_begin = time.perf_counter_ns()  # Start counting time
        for j, window in enumerate(windows):
            setattr(datapoints[j], attr, feature_func(window))

        # Determine how long it took to calculate a specific feature for all windows
        feature_timespan = time.perf_counter_ns() - time_begin
        durations[attr] = feature_timespan

    return datapoints, durations


# Writes a list of DataPoints to file.
# The file name and directory depends on the parameters.
def write_datapoints_csv(datapoints, period_ms, shuffle, stride_ms, impersonation_split, dos_type, set_type):
    csv_path, dir = get_dataset_path(period_ms, shuffle, stride_ms, impersonation_split, dos_type, set_type)

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


# Constructs a list of DataPoints based on parameters.
# 'period_ms' determines the duration of the time window used to create each DataPoint.
# 'stride_ms' determines how little of the previous time window may be used to create the next DataPoint.
# 'shuffle' dictates whether the list of DataPoints is to be randomized.
# 'impersonation_split' dictates whether the raw impersonation datasets,
# should be separated in attack free and attack affected data.
# 'dos_type' determines which DoS dataset should be used: 'original', 'modified'.
#
# Splits the list of DataPoints into two lists:
#   - a training set containing 80% of points.
#   - a test set containing 20% of points.
#
# If this function is to be used from another file,
# all code must be wrapped in an __name__ == '__main__' check if used on a Windows system.
def get_mixed_datasets(period_ms=100, shuffle=True, stride_ms=100, impersonation_split=True, dos_type='original'):
    # load messages and remove time offsets
    attack_free_messages1 = neutralize_offset(datareader_csv.load_attack_free1())
    attack_free_messages2 = neutralize_offset(datareader_csv.load_attack_free2())
    fuzzy_messages = neutralize_offset(datareader_csv.load_fuzzy())
    imp_messages1 = neutralize_offset(datareader_csv.load_impersonation_1())
    imp_messages2 = neutralize_offset(datareader_csv.load_impersonation_2())
    imp_messages3 = neutralize_offset(datareader_csv.load_impersonation_3())
    dos_messages = neutralize_offset(datareader_csv.load_dos() if dos_type == 'original' else
                                     datareader_csv.load_modified_dos())

    # label raw datasets
    raw_msgs = [
        (attack_free_messages1, "normal", "attack_free_1"),
        (attack_free_messages2, "normal", "attack_free_2"),
        (dos_messages, "dos", "dos"),
        (fuzzy_messages, "fuzzy", "fuzzy")]

    if impersonation_split:
        raw_msgs += [
            (imp_messages1[0:517000], "normal", "impersonation_normal_1"),
            (imp_messages1[517000:], "impersonation", "impersonation_attack_1"),
            (imp_messages2[0:330000], "normal", "impersonation_normal_2"),
            (imp_messages2[330000:], "impersonation", "impersonation_attack_2"),
            (imp_messages3[0:534000], "normal", "impersonation_normal_3"),
            (imp_messages3[534000:], "impersonation", "impersonation_attack_3")]
    else:
        raw_msgs += [
            (imp_messages1, "impersonation", "impersonation_1"),
            (imp_messages2, "impersonation", "impersonation_2"),
            (imp_messages3, "impersonation", "impersonation_3")
        ]

    datasets = []
    feature_durations_list = []

    # create DataPoints in parallel.
    with conf.ProcessPoolExecutor() as executor:
        futures = {executor.submit(messages_to_datapoints,
                                   tup[0],
                                   period_ms,
                                   tup[1],
                                   stride_ms,
                                   tup[2]) for tup in raw_msgs}

        for future in conf.as_completed(futures):
            datasets.append(future.result()[0])
            feature_durations_list.append(future.result()[1])

    offset = 0
    points = []

    # collapse resulting lists of DataPoints into a single list of continuous timestamps
    for dataset in datasets:
        time_low = dataset[0].time_ms
        points += [offset_datapoint(point, offset - time_low) for point in dataset]
        offset = points[len(points) - 1].time_ms

    feature_durations = {}

    # Collapse resulting feature duration dicts into a single duration dict
    for attr in dp.datapoint_attributes:
        feature_durations[attr] = 0
        for durations in feature_durations_list:
            feature_durations[attr] += durations[attr]

        # Average feature duration
        feature_durations[attr] /= len(points)

    # split the list of DataPoint into training (80%) and test (20%) sets
    training, test = train_test_split(points, shuffle=shuffle, train_size=0.8, test_size=0.2, random_state=2019)

    return training, test, feature_durations


# Increments the timestamp of input DataPoint by the input offset and returns the DataPoint
def offset_datapoint(point, offset):
    point.time_ms += offset

    return point


# Returns the file and directory paths associated with input argument combination.
def get_dataset_path(period_ms, shuffle, stride_ms, impersonation_split, dos_type, set_type):
    imp_name = "imp_split" if impersonation_split else "imp_full"
    shuffle_name = "shuffled" if shuffle else "normal"
    name = f"mixed_{set_type}_{period_ms}ms_{stride_ms}ms_{shuffle_name}"
    dir = f"data/feature/{imp_name}/{dos_type}/"

    return dir + name + ".csv", dir


# Returns the training and test sets associated with input argument combination.
# If the datasets do not exist, they are created and saved in the process.
def load_or_create_datasets(period_ms=100, shuffle=True, stride_ms=100,
                            impersonation_split=True, dos_type='original', force_create=False):
    training_name, _ = get_dataset_path(period_ms, shuffle, stride_ms, impersonation_split, dos_type, 'training')
    test_name, _ = get_dataset_path(period_ms, shuffle, stride_ms, impersonation_split, dos_type, 'test')

    # load the datasets if they exist.
    if os.path.exists(training_name) and os.path.exists(test_name) and not force_create:
        training_set = datareader_csv.load_idpoints(training_name)
        test_set = datareader_csv.load_idpoints(test_name)
        feature_durations = {}  # TODO load feature_durations
    else:
        # create and save the datasets otherwise.
        training_set, test_set, feature_durations = get_mixed_datasets(period_ms, shuffle, stride_ms, impersonation_split, dos_type)
        write_datapoints_csv(training_set, period_ms, shuffle, stride_ms, impersonation_split, dos_type, 'training')
        write_datapoints_csv(test_set, period_ms, shuffle, stride_ms, impersonation_split, dos_type, 'test')
        # TODO: write feature_durations

    return training_set, test_set, feature_durations

