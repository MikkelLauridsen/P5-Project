import datareader_csv
from recordclass import dataobject
import os
import csv
import math


class IDPoint(dataobject):
    time_ms: int
    is_injected: bool
    mean_id_interval: float
    variance_id_frequency: float
    num_id_transitions: int
    num_ids: int
    num_msgs: int

    def __str__(self):
        return f"time_ms: {self.time_ms} injected: {self.is_injected} mean interval: {self.mean_id_interval} frequency variance: " \
            f"{self.variance_id_frequency} transitions: {self.num_id_transitions} ids: {self.num_ids} msgs: {self.num_msgs}"


# Finds and returns the mean ID interval,
# where an ID interval is the time period between two messages of the same ID.
def calculate_mean_id_interval(messages):
    intervals = []
    last_seen_timestamps = {}

    for message in messages:
        if message.id in last_seen_timestamps:
            intervals.append(message.timestamp - last_seen_timestamps[message.id])

        last_seen_timestamps[message.id] = message.timestamp

    return intervals[math.floor(len(intervals) / 2)]


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

    return max(values) - min(values)


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

    return IDPoint(time_ms, is_injected, mean_id_interval, variance_id_frequency, num_id_transitions, num_ids, num_msgs)


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


# writes input IDPoint to file,
# using the input writer.
def write_idpoint_csv(idpoint, datafile_writer):
    datafile_writer.writerow([
        str(idpoint.time_ms), str(idpoint.is_injected), str(idpoint.mean_id_interval),
        str(idpoint.variance_id_frequency), str(idpoint.num_id_transitions),
        str(idpoint.num_ids), str(idpoint.num_msgs)])


# writes a list of IDPoints to file.
def write_idpoints_csv(idpoints, period_ms, name):
    # Creating a csv path for the new file using the corresponding csv file currently loaded from.
    dir = "idpoint_dataset/"
    csv_path = dir + name + "_" + str(len(idpoints)) + "_" + str(period_ms) + "ms.csv"

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(csv_path, "w", newline="") as datafile:
        datafile_writer = csv.writer(datafile, delimiter=";")

        # Writing the header.
        datafile_writer.writerow([
            "time_ms", "is_injected", "mean_id_interval", "variance_id_frequency",
            "num_id_transitions", "num_ids", "num_msgs"])

        for idpoint in idpoints:
            write_idpoint_csv(idpoint, datafile_writer)


# joins two lists of messages,
# by offsetting the timestamps of messages in the second list.
def concat_messages(msgs1, msgs2):
    offset = msgs1[len(msgs1) - 1].timestamp

    for msg in msgs2:
        msg.timestamp += offset

    return msgs1 + msgs2


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
        messages_to_idpoints(attack_free_messages1, period_ms, False),
        messages_to_idpoints(attack_free_messages2, period_ms, False),
        messages_to_idpoints(dos_messages, period_ms, True),
        messages_to_idpoints(fuzzy_messages[0:450000], period_ms, False),
        messages_to_idpoints(fuzzy_messages[450000:], period_ms, True),
        messages_to_idpoints(imp_messages1[0:517000], period_ms, False),
        messages_to_idpoints(imp_messages1[517000:], period_ms, True),
        messages_to_idpoints(imp_messages2[0:330000], period_ms, False),
        messages_to_idpoints(imp_messages2[330000:], period_ms, True),
        messages_to_idpoints(imp_messages3[0:534000], period_ms, False),
        messages_to_idpoints(imp_messages3[534000:], period_ms, True)]

    for set in datasets:
        training_high = math.floor(len(set) * 0.70)
        validation_high = math.floor(len(set) * 0.85)

        training += set[0:training_high]
        validation += set[training_high:validation_high]
        test += set[validation_high:]

    return (training, validation, test)


(training_set, validation_set, test_set) = get_mixed_datasets(100)

write_idpoints_csv(training_set, 100, "mixed_training")
write_idpoints_csv(validation_set, 100, "mixed_validation")
write_idpoints_csv(test_set, 100, "mixed_test")


# attack_free_messages = concat_messages(datareader_csv.load_attack_free1(), datareader_csv.load_attack_free2())
# attack_free_idpoints = messages_to_idpoints(attack_free_messages, 100, False)
# write_idpoints_csv(attack_free_idpoints, 100, "attack_free_full")

# dos_idpoints = messages_to_idpoints(datareader_csv.load_dos(), 100, True)
# write_idpoints_csv(dos_idpoints, 100, "dos_full")

# fuzzy_idpoints = messages_to_idpoints(datareader_csv.load_fuzzy(0, 450000), 100, False) + messages_to_idpoints(datareader_csv.load_fuzzy(450000), 100, True)
# write_idpoints_csv(fuzzy_idpoints, 100, "fuzzy_full")

# imp_messages1 = neutralize_offset(datareader_csv.load_impersonation_1())
# imp_idpoints1 = messages_to_idpoints(imp_messages1[0:517000], 100, False) + messages_to_idpoints(imp_messages1[517000:], 100, True)
# imp_messages2 = neutralize_offset(datareader_csv.load_impersonation_2())
# imp_idpoints2 = messages_to_idpoints(imp_messages2[0:330000], 100, False) + messages_to_idpoints(imp_messages2[330000:], 100, True)
# imp_messages3 = neutralize_offset(datareader_csv.load_impersonation_3())
# imp_idpoints3 = messages_to_idpoints(imp_messages3[0:534000], 100, False) + messages_to_idpoints(imp_messages3[534000:], 100, True)
# imp_idpoints = concat_idpoints(concat_idpoints(imp_idpoints1, imp_idpoints2), imp_idpoints3)
# write_idpoints_csv(imp_idpoints, 100, "impersonation_full")

