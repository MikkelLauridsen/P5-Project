"""
A collection of functions that analyse different aspects of the data. This analysis allows for decision making through
increased information and supports both the report and the implementation.

Aspects being analysed:

For finding the correct remote frame removal method:
Mean time between normal messages.
Mean time between split messages.

For finding the new attack-free/impersonation-attack split after remote frame removal:
Sum of removed time intervals.
Index of message before time cut-off.
"""
import numpy as np
from datareader_csv import load_messages


def get_mean_time_between_normal_messages(messages):
    """
    Finds the mean time between normal message neighbours, ignoring remote frames and remote frame responses.

    :param messages: A list of CAN bus Message objects.
    :param remote_frame_and_response_indices: A list of indices specifying where to find remote frames and responses.
    :return: The mean time between normal message pairs.
    """
    remote_frame_and_response_indices = __get_remote_frame_and_response_indices(messages)

    # The times between two normal messages.
    times_between_messages = []

    # Changing the list to a set to improve the time complexity of the "in" operation.
    remote_frame_and_response_indices = set(remote_frame_and_response_indices)

    # Using the list of non-normal indices to find the time between all neighbour pairs of normal messages.
    for i in range(len(messages) - 1):
        if i not in remote_frame_and_response_indices and i + 1 not in remote_frame_and_response_indices:
            times_between_messages.append(messages[i + 1].timestamp - messages[i].timestamp)

    return np.mean(times_between_messages)


def get_mean_time_between_split_messages(messages):
    """
    Returns the mean time between message pairs that have a remote frame or remote frame response between them.

    :param messages: A list of CAN bus Message objects.
    :param remote_frame_and_response_indices: A list of indices specifying where to find remote frames and responses.
    :return: 0 if there are no remote frames and responses and the mean of split normal message pairs if there is.
    """
    remote_frame_and_response_indices = __get_remote_frame_and_response_indices(messages)

    times_between_messages = []

    # Using the list of non-normal indices to find all neighbours of remote frames and remote frame responses.
    for index in remote_frame_and_response_indices:
        times_between_messages.append(messages[index + 1].timestamp - messages[index - 1].timestamp)

    return np.mean(times_between_messages) if len(times_between_messages) > 0 else 0


def get_sum_of_removed_intervals(messages, time):
    """
    Finds the sum of the intervals that would be removed in the list of messages before the given timestamp.

    :param messages: A list of CAN bus Message objects.
    :param time: The time specifying how many seconds of the file the sum should account for.
    :return: The sum of the intervals that would be removed.
    """
    removed_intervals = []
    remote_frame_and_response_indices = __get_remote_frame_and_response_indices(messages)

    # Getting the timestamp that specifies the cutoff.
    timestamp = messages[0].timestamp + time

    # Going through each remote frame and response with a timestamp below the specified limit.
    for index in remote_frame_and_response_indices:
        if messages[index].timestamp > timestamp:
            break
        else:
            removed_intervals.append(messages[index].timestamp - messages[index - 1].timestamp)

    # Return the sum
    return sum(removed_intervals)


def get_index_before_time(messages, time):
    """Finds the index of the message that is immediately before the point where the given time has passed."""
    # Getting the timestamp that specifies the cutoff.
    timestamp = messages[0].timestamp + time

    for index, message in enumerate(messages):
        if message.timestamp > timestamp:
            return index - 1


def __get_remote_frame_and_response_indices(messages):
    # Finding the indices of every remote frame and remote frame response in the given list of messages.

    latest_remote_frame = None
    latest_remote_frame_index = None

    remote_frame_or_response_indices = []

    # Finding all the indices of the remote frames and remote frame responses
    for i in range(len(messages)):
        # If the message is a remote frame.
        if messages[i].rtr == 0b100:
            latest_remote_frame = messages[i]
            latest_remote_frame_index = i
            remote_frame_or_response_indices.append(i)
        # If it is a remote frame response.
        elif latest_remote_frame is not None and \
                messages[i].id == latest_remote_frame.id and i - latest_remote_frame_index < 7:
            latest_remote_frame = None
            latest_remote_frame_index = None
            remote_frame_or_response_indices.append(i)

    return remote_frame_or_response_indices


def analyze_data():
    """
    Calls the information gathering functions in a centralized manner

    :return: A dictionary where the keys are descriptions of the information in the values
    """
    attack_free_1 = load_messages("data/csv/Attack_free_dataset.csv", verbose=True)

    impersonation_1 = load_messages("data/csv/170907_impersonation.csv", verbose=True)
    impersonation_2 = load_messages("data/csv/170907_impersonation_2.csv", verbose=True)
    impersonation_3 = load_messages("data/csv/Impersonation_attack_dataset.csv", verbose=True)

    information = {
        "Mean time between normal messages":
            get_mean_time_between_normal_messages(attack_free_1),
        "Mean time between split messages":
            get_mean_time_between_split_messages(attack_free_1),
        "Sum of removed intervals in '170907_impersonation.csv'":
            get_sum_of_removed_intervals(impersonation_1, 250),
        "Sum of removed intervals in '170907_impersonation_2.csv'":
            get_sum_of_removed_intervals(impersonation_2, 250),
        "Sum of removed intervals in 'Impersonation_attack_dataset.csv'":
            get_sum_of_removed_intervals(impersonation_3, 250),
        "Index of split in '170907_impersonation.csv'":
            get_index_before_time(impersonation_1, 250 - 23.434627056121826),
        "Index of split in '170907_impersonation_2.csv'":
            get_index_before_time(impersonation_2, 250 - 20.980855226516724),
        "Index of split in 'Impersonation_attack_dataset.csv'":
            get_index_before_time(impersonation_3, 250 - 2.1056361198425293)
    }

    return information


if __name__ == "__main__":
    information = analyze_data()
    for key, value in information.items():
        print(f"{key}: {value}")

    # Without removing remote frames: 0.00042314722179202243
    # Removing remote frames without closing the holes: 0.00047852379197671924
    # Removing remote frames and closing the holes: 0.0004403009096925493
    # Average time between split messages: 0.0009379362399745852
