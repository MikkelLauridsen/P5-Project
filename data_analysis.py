"""Functions for analysing aspects of the data. The analysis is used to make decision about data manipulation."""
import numpy as np
import datareader_csv


def get_mean_time_between_normal_messages(messages):
    """Returns the mean time between normal message neighbours, ignoring remote frames and remote frame responses."""

    # The times between two normal messages.
    times_between_messages = []

    latest_remote_frame = None
    latest_remote_frame_index = None

    remote_frame_or_response_indices = []

    # Finding all the indices of the remote frames and remote frame responses
    for i in range(len(messages)):
        # If the next message is a remote frame.
        if messages[i] == 0b100:
            latest_remote_frame = messages[i]
            latest_remote_frame_index = i
            remote_frame_or_response_indices.append(i)
        # If it is a remote frame response.
        elif latest_remote_frame is not None and \
                messages[i].id == latest_remote_frame.id and i - latest_remote_frame_index < 7:
            latest_remote_frame = None
            latest_remote_frame_index = None
            remote_frame_or_response_indices.append(i)

    # Using the list of non-normal indexes to find the time between all neighbour pairs of normal messages.
    for i in range(len(messages) - 1):
        if i not in remote_frame_or_response_indices and i + 1 not in remote_frame_or_response_indices:
            times_between_messages.append(messages[i + 1].timestamp - messages[i].timestamp)

    return np.mean(times_between_messages)


if __name__ == "__main__":
    messages = datareader_csv.load_attack_free1()

    mean_time_between_normal_messages = get_mean_time_between_normal_messages(messages)

    print(f"Average time between normal messages: {mean_time_between_normal_messages}")

    # Without removing remote frames: 0.0004379132395288759
    # Removing remote frames without closing the holes: 0.00047852379197671924
    # Removing remote frames and closing the holes: 0.0004403009096925493
