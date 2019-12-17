"""
Running this file takes all csv files in the "csv" directory, manipulates them and puts the result in the
"manipulated" directory.

Manipulations:
DLC: The DLC for every message in the file "Attack_free_dataset2.csv" is changed to 8.
Remote frames: Remote frames and remote frame responses are removed from every file.
Modified DoS: The data field of every DoS injected message is changed to represent a more realistic attack.
"""
import pandas as pd
import random
import os
import datareader_csv
import csv
import message as msg


def manipulate_dlc(source, target, new_dlc):
    """
    Replaces every DLC in the specified file with the given new DLC

    :param source: The source file that is going to be changed.
    :param target: The target file where the changes are going to be saved.
    :param new_dlc: The new DLC value that every message in the source file will get.
    :return:
    """
    # Getting the data from the filepath.
    data = pd.read_csv(source)

    # Replacing all values in the dlc column with the specified dlc.
    data = data.assign(dlc=new_dlc)

    data.to_csv(target, columns=["timestamp", "id", "rtr", "dlc", "data"], index=False)


def manipulate_dos_data_field(source, target):
    """
    Replaces the data field of DoS injected messages with a more realistic data field.

    :param source: The source file that is going to be changed.
    :param target: The target file where the changes are going to be saved.
    :return:
    """
    # Getting the data from the filepath.
    data = pd.read_csv(source)

    # Getting the "normal" data fields, hereby meaning non-attack messages that are not remote frames.
    normal_data_fields = __get_normal_data_fields(data)

    data = __replace_dos_messages(data, normal_data_fields)

    # Converting the manipulated dataset back into a csv file.
    data.to_csv(target, columns=["timestamp", "id", "rtr", "dlc", "data"], index=False)


def __get_normal_data_fields(df):
    # Goes through a dataframe and returns list containing the data fields for normal messages. Only works for DoS.

    # Getting all attack free messages in the file.
    df = df.drop(df[df.id == 0].index)

    # Removing remote frames.
    df = df.dropna()

    return df["data"].tolist()


def __replace_dos_messages(df, normal_data_fields):
    # Going through all DoS injected messages and replacing the data field with a random normal data field.
    # Also updates the data length control column for the message.

    for row in df.itertuples():
        if row.id == 0:
            # Setting the seed with "Random(row.Index)" so it is consistent across multiple program executions.
            df.at[row.Index, "data"] = random.Random(row.Index).choice(normal_data_fields)
            df.at[row.Index, "dlc"] = len(df.at[row.Index, "data"].split(" "))

    return df


def manipulate_remote_frames(source, target):
    """
    Removes remote frames and remote frame responses and closes the time gaps created by this removal.

    :param source: The source file that is going to be changed.
    :param target: The target file where the changes are going to be saved.
    :return:
    """
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
    # Removes remote frames and their corresponding responses. It is assumed that responses arrive withing 7 messages.

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
            message.timestamp -= current_offset
            i += 1


if __name__ == "__main__":
    os.chdir("..")

    source_dir = "data/csv/"
    target_dir = "data/manipulated/"

    # Creating the target directory if it does not exist.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    manipulate_dos_data_field(f"{source_dir}DoS_attack_dataset.csv", f"{target_dir}DoS_manipulated.csv")

    # Going through each file in the source dir.
    for filename in os.listdir(source_dir):
        manipulate_remote_frames(f"{source_dir}{filename}", f"{target_dir}{filename}")

    manipulate_remote_frames("data/manipulated/DoS_manipulated.csv", f"{target_dir}DoS_manipulated.csv")
    manipulate_dlc("data/manipulated/Attack_free_dataset2.csv", f"{target_dir}Attack_free_dataset2.csv", 8)
