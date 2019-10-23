"""Functions for manipulating data."""

import pandas as pd
import random
import os
import datareader_csv
import csv
import message as msg


def manipulate_dos(filepath, target_dir):
    """Replaces DoS injection messages with more realistic attacks and outputs the changed file in the target dir."""
    # Getting the data from the filepath.
    data = pd.read_csv(filepath)

    # Getting the "normal" data fields, hereby meaning non-attack messages that are not remote frames.
    normal_data_fields = __get_normal_data_fields(data)

    data = __replace_dos_messages(data, normal_data_fields)

    # Converting the manipulated dataset back into a csv file.
    data.to_csv(target_dir, columns=["timestamp", "id", "rtr", "dlc", "data"], index=False)


# Goes through a dataframe and returns list containing the data fields for normal messages. Only works for DoS.
def __get_normal_data_fields(df):
    # Getting all attack free messages in the file.
    df = df.drop(df[df.id == 0].index)

    # Removing remote frames.
    df = df.dropna()

    return df["data"].tolist()


# Going through all DoS injected messages and replacing the data field with a random normal data field.
# Also updates the data length control column for the message.
def __replace_dos_messages(df, normal_data_fields):
    for row in df.itertuples():
        if row.id == 0:
            # Setting the seed with "Random(row.Index)" so it is consistent across multiple program executions.
            df.at[row.Index, "data"] = random.Random(row.Index).choice(normal_data_fields)
            df.at[row.Index, "dlc"] = len(df.at[row.Index, "data"].split(" "))

    return df


def manipulate_remote_frames(source, target):
    print(f"Removing remote frames and corresponding responses from {source}")
    messages = datareader_csv.load_messages(source)
    __remove_remote_frames(messages)

    with open(target, "w", newline="") as datafile:
        datafile_writer = csv.writer(datafile, delimiter=",")

        # Writing the header.
        datafile_writer.writerow(msg.message_attributes)

        for message in messages:
            datafile_writer.writerow(msg.get_csv_row(message))


def __remove_remote_frames(messages):
    """Removes remote frames and their corresponding responses.
    It is assumed that responses arrive withing 7 messages."""
    latest_remote_frame = None
    latest_remote_frame_index = None
    current_offset = 0

    # Iterate messages
    i = 0
    while i < len(messages):
        message = messages[i]

        # Remove remote frames
        if message.rtr == 0b100:
            latest_remote_frame = message
            latest_remote_frame_index = i
            current_offset = message.timestamp - messages[i - 1].timestamp
            del messages[i]
        # Remove responses to remote frames. We assume responses arrive within 7 messages.
        elif latest_remote_frame is not None and \
                message.id == latest_remote_frame.id and i - latest_remote_frame_index < 7:
            latest_remote_frame = None
            latest_remote_frame_index = None
            current_offset = message.timestamp - messages[i - 1].timestamp
            del messages[i]
        else:
            # Continue to next message
            # message.timestamp -= current_offset  # Removed until after meeting
            i += 1


if __name__ == "__main__":
    source_dir = "data/data_csv/"
    target_dir = "data/manipulated_data/"

    # Creating the target directory if it does not exist.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    manipulate_dos(source_dir + "DoS_attack_dataset.csv", target_dir + "DoS_manipulated.csv")

    manipulate_remote_frames(source_dir + "Attack_free_dataset.csv", target_dir + "Attack_free_dataset.csv")
    manipulate_remote_frames(source_dir + "Attack_free_dataset2.csv", target_dir + "Attack_free_dataset2.csv")
    manipulate_remote_frames(source_dir + "DoS_attack_dataset.csv", target_dir + "DoS_attack_dataset.csv")
    manipulate_remote_frames(source_dir + "Fuzzy_attack_dataset.csv", target_dir + "Fuzzy_attack_dataset.csv")
    manipulate_remote_frames(source_dir + "Impersonation_attack_dataset.csv", target_dir + "Impersonation_attack_dataset.csv")
    manipulate_remote_frames(source_dir + "170907_impersonation.csv", target_dir + "170907_impersonation.csv")
    manipulate_remote_frames(source_dir + "170907_impersonation_2.csv", target_dir + "170907_impersonation_2.csv")
