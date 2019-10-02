import idpoint as idp
import datareader_csv
import os
import csv
import math
import functools as ft


def __calculate_kurtosis(values):
    n = len(values)
    avg = math.fsum(values) / n
    s2 = ft.reduce(lambda x: (x - avg) ** 2, values, 0)
    s4 = ft.reduce(lambda x: (x - avg) ** 4, values, 0)
    m2 = s2 / n
    m4 = s4 / n

    return m4 / (m2 ** 2)


def __calculate_variance(values):
    avg_freq = math.fsum(values) / len(values)
    variance = 0

    for freq in values:
        deviation = freq - avg_freq
        variance += deviation * deviation

    return (1.0 / len(values)) * variance


def __calculate_bit_count(message):
    if message.dlc == 0:
        return 0

    count = 0

    for byte in message.data:
        for i in range(8):
            if byte & (0b10000000 >> i):
                count += 1

    return count


def calculate_mean_probability_bits(messages):
    bits = [0 for i in range(64)]

    for message in messages:
        if message.dlc != 0:
            for i in range(len(message.data)):
                for j in range(8):
                    if message.data[i] & (0b10000000 >> j):
                        bits[i*8+j] += 1

    for i in range(64):
        bits[i] = bits[i] / len(messages)

    return math.fsum(bits) / 64.0


def calculate_mean_id_intervals_variance(messages):
    id_timestamp_intervals = {}
    last_seen_timestamps = {}

    for message in messages:
        if message.id in last_seen_timestamps:
            interval = message.timestamp - last_seen_timestamps[message.id]
            id_timestamp_intervals.setdefault(message.id, [])
            id_timestamp_intervals[message.id].append(interval)

        last_seen_timestamps[message.id] = message.timestamp

    intervals_variances = []
    for intervals in id_timestamp_intervals.values():
        intervals_variances.append(__calculate_variance(intervals))

    return math.fsum(intervals_variances) / len(intervals_variances)


def calculate_mean_data_bit_count(messages):
    counts = []

    for message in messages:
        counts += [__calculate_bit_count(message)]

    return math.fsum(counts) / len(counts)


def calculate_variance_data_bit_count(messages):
    counts = []

    for message in messages:
        counts += [__calculate_bit_count(message)]

    return __calculate_variance(counts)


def calculate_mean_variance_data_bit_count_id(messages):
    id_counts = {}

    for message in messages:
        if message.id in id_counts:
            id_counts[message.id] += [__calculate_bit_count(message)]
        else:
            id_counts[message.id] = [__calculate_bit_count(message)]

    variances = []

    for counts in id_counts.values():
        variances += [__calculate_variance(counts)]

    return math.fsum(variances) / len(variances)


# Finds and returns the mean ID interval,
# where an ID interval is the time period between two messages of the same ID.
def calculate_mean_id_interval(messages):
    intervals = []
    last_seen_timestamps = {}

    for message in messages:
        if message.id in last_seen_timestamps:
            intervals.append(message.timestamp - last_seen_timestamps[message.id])

        last_seen_timestamps[message.id] = message.timestamp

    return math.fsum(intervals) / len(intervals)


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


def calculate_req_to_res_time_variance(messages):
    intervals = []
    latest_remote_frame_timestamp = {}

    for message in messages:
        if message.add == 0b100:
            latest_remote_frame_timestamp[message.id] = message.timestamp
        elif message.add == 0b000 and latest_remote_frame_timestamp.get(message.id, None) is not None:
            intervals.append(message.timestamp - latest_remote_frame_timestamp[message.id])
            latest_remote_frame_timestamp[message.id] = None

    if len(intervals) == 0:
        return 0

    return __calculate_variance(intervals)


# Converts input 'messages' to an IDPoint object.
# 'is_injected' determines whether intrusion was conducted in 'messages'
def messages_to_idpoint(messages, is_injected):
    # this function may never be called with an empty list
    time_ms = messages[0].timestamp * 1000
    mean_id_interval = calculate_mean_id_interval(messages)
    variance_id_frequency = calculate_variance_id_frequency(messages)
    num_id_transitions = calculate_num_id_transitions(messages)
    num_ids = calculate_num_ids(messages)
    num_msgs = len(messages)
    mean_id_intervals_variance = calculate_mean_id_intervals_variance(messages)
    mean_data_bit_count = calculate_mean_data_bit_count(messages)
    variance_data_bit_count = calculate_variance_data_bit_count(messages)
    mean_variance_data_bit_count_id = calculate_mean_variance_data_bit_count_id(messages)
    mean_probability_bits = calculate_mean_probability_bits(messages)
    req_to_res_time_variance = calculate_req_to_res_time_variance(messages)

    return idp.IDPoint(time_ms, is_injected, mean_id_interval, variance_id_frequency,
                       num_id_transitions, num_ids, num_msgs, mean_id_intervals_variance,
                       mean_data_bit_count, variance_data_bit_count,
                       mean_variance_data_bit_count_id, mean_probability_bits, req_to_res_time_variance)


# Converts a list of messages to a list of IDPoints,
# where each point is comprised of 'messages' in 'period_ms' time interval.
# 'is_injected' determines whether intrusion was conducted in 'messages'
def messages_to_idpoints(messages, period_ms, is_injected):
    if len(messages) == 0:
        return []

    period_low = messages[0].timestamp
    id_low = 0
    idpoints = []

    for i in range(len(messages)):
        if (messages[i].timestamp - period_low) * 1000.0 > period_ms:
            idpoints.append(messages_to_idpoint(messages[id_low:i], is_injected))
            period_low = messages[i].timestamp
            id_low = i

    return idpoints


# writes a list of IDPoints to file.
def write_idpoints_csv(idpoints, period_ms, name):
    # Creating a csv path for the new file using the corresponding csv file currently loaded from.
    dir = "data/idpoint_dataset/"
    csv_path = dir + name + "_" + str(len(idpoints)) + "_" + str(period_ms) + "ms.csv"

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(csv_path, "w", newline="") as datafile:
        datafile_writer = csv.writer(datafile, delimiter=",")

        # Writing the header.
        datafile_writer.writerow(idp.csv_header_row)

        for idpoint in idpoints:
            datafile_writer.writerow(idp.get_csv_row(idpoint))


# joins two lists of messages,
# by offsetting the timestamps of messages in the second list.
def concat_messages(msgs1, msgs2):
    offset = msgs1[len(msgs1) - 1].timestamp

    for msg in msgs2:
        msg.timestamp += offset

    return msgs1 + msgs2


# modifies input list of messages,
# such that the first message starts at time 0.
# returns the changed input list.
def neutralize_offset(messages):
    offset = messages[0].timestamp

    for message in messages:
        message.timestamp -= offset

    return messages


def concat_idpoints(idpoints1, idpoints2):
    offset = idpoints1[len(idpoints1) - 1].time_ms

    for idpoint in idpoints2:
        idpoint.time_ms += offset

    return idpoints1 + idpoints2


# returns a tuple containing:
#   - a training set comprised of 70% of the data
#   - a validation set comprised of 15% of the data
#   - a test set comprised of 15% of the data
def get_mixed_datasets(period_ms):
    training, validation, test = [], [], []

    attack_free_messages1 = neutralize_offset(datareader_csv.load_attack_free1())
    attack_free_messages2 = neutralize_offset(datareader_csv.load_attack_free2())
    dos_messages = neutralize_offset(datareader_csv.load_dos())
    fuzzy_messages = neutralize_offset(datareader_csv.load_fuzzy())
    imp_messages1 = neutralize_offset(datareader_csv.load_impersonation_1())
    imp_messages2 = neutralize_offset(datareader_csv.load_impersonation_2())
    imp_messages3 = neutralize_offset(datareader_csv.load_impersonation_3())

    datasets = [
        messages_to_idpoints(attack_free_messages1, period_ms, "normal"),
        messages_to_idpoints(attack_free_messages2, period_ms, "normal"),
        messages_to_idpoints(dos_messages, period_ms, "dos"),
        messages_to_idpoints(fuzzy_messages, period_ms, "fuzzy"),
        messages_to_idpoints(imp_messages1[0:517000], period_ms, "normal"),
        messages_to_idpoints(imp_messages1[517000:], period_ms, "impersonation"),
        messages_to_idpoints(imp_messages2[0:330000], period_ms, "normal"),
        messages_to_idpoints(imp_messages2[330000:], period_ms, "impersonation"),
        messages_to_idpoints(imp_messages3[0:534000], period_ms, "normal"),
        messages_to_idpoints(imp_messages3[534000:], period_ms, "impersonation")]

    offset_training, offset_validation, offset_test = 0, 0, 0

    for set in datasets:
        training_high = math.floor(len(set) * 0.70)
        validation_high = math.floor(len(set) * 0.85)

        training_low = set[0].time_ms
        validation_low = set[training_high].time_ms
        test_low = set[validation_high].time_ms

        training += [offset_idpoint(idp, offset_training - training_low) for idp in set[0:training_high]]
        validation += [offset_idpoint(idp, offset_validation - validation_low) for idp in set[training_high:validation_high]]
        test += [offset_idpoint(idp, offset_test - test_low) for idp in set[validation_high:]]

        offset_training = training[len(training) - 1].time_ms
        offset_validation = validation[len(validation) - 1].time_ms
        offset_test = test[len(test) - 1].time_ms

    return training, validation, test


def offset_idpoint(idp, offset):
    idp.time_ms += offset

    return idp


if __name__ == "__main__":
    training_set, validation_set, test_set = get_mixed_datasets(100)

    write_idpoints_csv(training_set, 100, "mixed_training")
    write_idpoints_csv(validation_set, 100, "mixed_validation")
    write_idpoints_csv(test_set, 100, "mixed_test")

    #imp_messages1 = neutralize_offset(datareader_csv.load_impersonation_1())
    #imp_idpoints1 = messages_to_idpoints(imp_messages1[0:517000], 100, False) + messages_to_idpoints(imp_messages1[517000:], 100, True)
    #imp_messages2 = neutralize_offset(datareader_csv.load_impersonation_2())
    #imp_idpoints2 = messages_to_idpoints(imp_messages2[0:330000], 100, False) + messages_to_idpoints(imp_messages2[330000:], 100, True)
    #imp_messages3 = neutralize_offset(datareader_csv.load_impersonation_3())
    #imp_idpoints3 = messages_to_idpoints(imp_messages3[0:534000], 100, False) + messages_to_idpoints(imp_messages3[534000:], 100, True)
    #imp_idpoints = concat_idpoints(concat_idpoints(imp_idpoints1, imp_idpoints2), imp_idpoints3)
    #write_idpoints_csv(imp_idpoints, 100, "impersonation_full")
