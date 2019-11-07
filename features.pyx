from cpython cimport array
import math
import functools as ft
import datapoint as dp
import time
import array


cdef array.array to_array(values):
    return array.array("d", values)


# Calculates the variance of elements in input list.
# The equation for population variance is used.
# Defaults to 0 if the input list has less than 2 elements.
cdef double __calculate_variance(array.array values):
    cdef double avg_freq, variance, deviation, val

    avg_freq = 0
    for val in values:
        avg_freq += val
    avg_freq /= len(values)

    variance = 0

    for freq in values:
        deviation = freq - avg_freq
        variance += deviation * deviation

    return (1.0 / len(values)) * variance


# Calculates the kurtosis of the values in input list.
# Bock's kurtosis coefficient is used, which means the kurtosis of the normal distribution is 3.
# The kurtosis also defaults to 3 if input list has less than two elements.
cdef double __calculate_kurtosis(array.array values):
    cdef double n, avg, s2, s4, m2, m4
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


# Calculates the skewness of the values in input list.
# Pearson's second skewness coefficient is used as equation.
# If the list has less than two elements, the skewness will default to 0.
cdef double __calculate_skewness(array.array values):
    cdef double mean, median, variance
    if len(values) == 0:
        return 0

    list(values).sort()  # sort values, such that the median can be found

    mean = math.fsum(values) / len(values)
    median = values[math.floor(len(values) / 2)]
    variance = __calculate_variance(values)

    if variance == 0:
        return 0

    return (3 * (mean - median)) / math.sqrt(variance)


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


# For each unique ID in input list of messages,
# constructs a list of time periods between two messages with this ID.
# Afterwards, the variance each list is found.
# Finally, the skewness of these variances is calculated and returned.
def calculate_skewness_id_interval_variances(messages):
    id_timestamp_intervals = __find_id_intervals(messages)

    intervals_variances = []
    for intervals in id_timestamp_intervals.values():
        intervals_variances.append(__calculate_variance(to_array(intervals)))

    return __calculate_skewness(to_array(intervals_variances))


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
        variances.append(__calculate_variance(to_array(counts)))

    return __calculate_kurtosis(to_array(variances))


# For each unique ID in input list of messages,
# constructs a list of time periods between messages of this ID.
# Afterwards, calculates the variance of elements in each list, separately.
# Finally, calculates and returns the mean of these variances.
# Returned value defaults to 0 if no intervals were present in the time window.
def calculate_mean_id_intervals_variance(messages):
    id_timestamp_intervals = __find_id_intervals(messages)

    intervals_variances = []
    for intervals in id_timestamp_intervals.values():
        intervals_variances.append(__calculate_variance(to_array(intervals)))

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

    return __calculate_variance(to_array(counts))


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
        variances.append(__calculate_variance(to_array(counts)))

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

    return __calculate_variance(to_array(frequencies.values()))


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


# Constructs a list of time periods between messages of the same ID.
# Afterwards, calculates and returns the kurtosis of these time periods.
def calculate_kurtosis_id_interval(messages):
    intervals = []
    last_seen_timestamps = {}

    for message in messages:
        if message.id in last_seen_timestamps:
            intervals.append(message.timestamp - last_seen_timestamps[message.id])

        last_seen_timestamps[message.id] = message.timestamp

    return __calculate_kurtosis(to_array(intervals))


# Calculates the frequency of each ID in input list of messages.
# Afterwards, calculates and returns the skewness of these frequencies.
def calculate_skewness_id_frequency(messages):
    frequencies = {}

    for message in messages:
        if message.id not in frequencies:
            frequencies[message.id] = 1
        else:
            frequencies[message.id] += 1

    #values = list(frequencies.values())

    return __calculate_skewness(to_array(frequencies.values()))


# Calculates the frequency of each ID in input list of messages.
# Afterwards, calculates and returns the kurtosis of these frequencies.
def calculate_kurtosis_id_frequency(messages):
    frequencies = {}

    for message in messages:
        if message.id not in frequencies:
            frequencies[message.id] = 1
        else:
            frequencies[message.id] += 1

    #values = frequencies.values()

    return __calculate_kurtosis(to_array(frequencies.values()))


# For each unique ID in input list of messages,
# constructs a list of time periods between messages with this ID.
# Afterwards, calculates the means of elements of these lists, separately.
# Finally, calculates and returns the kurtosis of these means.
def calculate_kurtosis_mean_id_intervals(array.array messages):
    id_timestamp_intervals = __find_id_intervals(messages)

    interval_means = []
    for intervals in id_timestamp_intervals.values():
        interval_means.append(sum(intervals) / len(intervals))

    return __calculate_kurtosis(to_array(interval_means))

# Calculates a list of datapoints from a list of windows (a window being a list of messages)
# 'class_label' determines whether intrusion was conducted in 'messages'
# this function may never be called with an empty list
def windows_to_datapoints(windows, class_label, name):
    # maps a function to an attribute. The function must accept a list of messages.
    # missing mappings are allowed, and will give the feature a value of 0
    attribute_function_mappings = {
        "time_ms": lambda msgs: msgs[0].timestamp * 1000,
        "class_label": lambda msgs: class_label,
        "mean_id_interval": calculate_mean_id_interval,
        "variance_id_frequency": calculate_variance_id_frequency,
        "num_id_transitions": calculate_num_id_transitions,
        "num_ids": calculate_num_ids,
        "num_msgs": len,
        "mean_id_intervals_variance": calculate_mean_id_intervals_variance,
        "mean_data_bit_count": calculate_mean_data_bit_count,
        "variance_data_bit_count": calculate_variance_data_bit_count,
        "mean_variance_data_bit_count_id": calculate_mean_variance_data_bit_count_id,
        "kurtosis_id_interval": calculate_kurtosis_id_interval,
        "kurtosis_id_frequency": calculate_kurtosis_id_frequency,
        "kurtosis_mean_id_intervals": calculate_kurtosis_mean_id_intervals,
        "kurtosis_variance_data_bit_count_id": calculate_kurtosis_variance_data_bit_count_id,
        "skewness_id_interval_variances": calculate_skewness_id_interval_variances,
        "skewness_id_frequency": calculate_skewness_id_frequency,
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