import datareader_csv
from recordclass import dataobject
import math


class IDPoint(dataobject):
    is_injected: bool
    mean_id_interval: float
    variance_id_frequency: float
    num_id_transitions: int
    num_ids: int
    num_msgs: int

    def __str__(self):
        return f"injected: {self.is_injected} mean interval: {self.mean_id_interval} frequency variance: " \
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
    mean_id_interval = calculate_mean_id_interval(messages)
    variance_id_frequency = calculate_variance_id_frequency(messages)
    num_id_transitions = calculate_num_id_transitions(messages)
    num_ids = calculate_num_ids(messages)
    num_msgs = len(messages)

    return IDPoint(is_injected, mean_id_interval, variance_id_frequency, num_id_transitions, num_ids, num_msgs)


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


messages = datareader_csv.load_data("data_csv/Attack_free_dataset.csv", 0, 1000000)
idpoints = messages_to_idpoints(messages, 100, False)

print(idpoints)
print(f"Num idpoints: {len(idpoints)}")