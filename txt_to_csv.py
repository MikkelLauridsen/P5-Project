"""
Running this file takes every txt file in the "raw" directory and turns them into equivalent csv files.
The resulting csv files are put into the "csv" directory.
"""
import os
import re

import pandas as pd


def __parse_message(message_str, pattern):
    # This function uses regular expressions to match the contents of a message to its specific parts.

    m = re.match(pattern, message_str)
    timestamp = float(m.group("timestamp"))
    id = int(m.group("id"), 16)
    dlc = int(m.group("dlc"))
    data = m.group("data")

    try:
        rtr = int(m.group("rtr"), 2)
    except IndexError:
        rtr = 0

    return [timestamp, id, rtr, dlc, data]


def __load_data(filepath, pattern, start=0):
    # Takes a filepath and a pattern for parsing rows, returns a pandas data frame containing the contents of the file.

    print(f"Started reading from {filepath}")
    data = []
    with open(filepath, "r") as fileobject:
        # Skipping the header if necessary
        if start == 1:
            next(fileobject)

        for index, row in enumerate(fileobject):
            if row != "":
                data.append(__parse_message(row, pattern))

            if index % 50000 == 0:
                print(f"Reading message: {index}")
    print("Completed")

    return pd.DataFrame(data)


pattern1 = r"Timestamp:( )*(?P<timestamp>.*)        ID: (?P<id>[0-9a-f]*)    (?P<rtr>[01]*)    "\
           r"DLC: (?P<dlc>[0-8])(    (?P<data>(([0-9a-f]+)( )?)*))?"
pattern2 = r"(?P<id>[0-9a-f]*)	(?P<dlc>[0-8])	(?P<data>(([0-9a-f]*)( )?)*)		( )*(?P<timestamp>.*)"


def txt_to_csv(start_dir, target_dir):
    """
    Taking all txt files in the start dir, converting them to csv files and putting them in the target dir.

    :param start_dir: The directory containing the txt files that should be converted to csv files.
    :param target_dir: The directory that is going to contain the newly generated csv files.
    :return:
    """
    # Creating the directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Creating a list containing all file paths to raw data and going through it.
    for path in [start_dir + i for i in os.listdir(start_dir)]:
        # The Attack_free_dataset2 file needs to be called with special conditions because of its format.
        if path == f"{start_dir}Attack_free_dataset2.txt":
            text_file = __load_data(path, pattern2, start=1)
        else:
            text_file = __load_data(path, pattern1)

        # Converting the data frame into a csv file and placing it in the target dir.
        text_file.to_csv(f"{path.replace(start_dir, target_dir)[:-3]}csv",
                         header=["timestamp", "id", "rtr", "dlc", "data"], index=False)


if __name__ == "__main__":
    txt_to_csv("data/raw/", "data/csv/")
