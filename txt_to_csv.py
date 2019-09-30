import re
import os
import csv

import message


def __parse_message(message_str, pattern):
    m = re.match(pattern, message_str)
    timestamp = float(m.group("timestamp"))
    id = int(m.group("id"), 16)
    dlc = int(m.group("dlc"))

    add = 0
    try:
        add = int(m.group("add"), 2)
    except IndexError:
        pass

    raw_data = None
    if dlc > 0:
        raw_data = m.group("data").split(" ")

    return message.Message(timestamp, id, add, dlc, raw_data)


def __load_data(filepath, pattern, start=0):
    data = []
    with open(filepath, "r") as fileobject:
        # Skipping the header if necessary
        if start == 1:
            next(fileobject)

        for index, row in enumerate(fileobject):
            if row != "":
                data.append(__parse_message(row, pattern))

            if index % 50000 == 0:
                print(index)
    return data


pattern1 = r"Timestamp:( )*(?P<timestamp>.*)        ID: (?P<id>[0-9a-f]*)    (?P<add>[01]*)    "\
           r"DLC: (?P<dlc>[0-8])(    (?P<data>(([0-9a-f]+)( )?)*))?"
pattern2 = r"(?P<id>[0-9a-f]*)	(?P<dlc>[0-8])	(?P<data>(([0-9a-f]*)( )?)*)		( )*(?P<timestamp>.*)"


# Returning a list containing all the different txt file paths in no particular order.
def __get_paths():
    return ["data/raw_data/Attack_free_dataset.txt", "data/raw_data/Attack_free_dataset2.txt",
            "data/raw_data/Impersonation_attack_dataset.txt", "data/raw_data/170907_impersonation.txt",
            "data/raw_data/170907_impersonation_2.txt", "data/raw_data/DoS_attack_dataset.txt",
            "data/raw_data/Fuzzy_attack_dataset.txt"]


def txt_to_csv():
    # Creating the directory if it does not exist
    if not os.path.exists("data\data_csv"):
        os.makedirs("data/data_csv")

    # Getting a list of links between each txt file function and their equivalent csv file.
    txt_file_paths = __get_paths()

    # Going through each link between txt file functions and csv files.
    for i in range(len(txt_file_paths)):
        # The attack free 2 raw_data set needs to be called with special conditions because of its format.
        if txt_file_paths[i] == "data/raw_data/Attack_free_dataset2.txt":
            text_file = __load_data(txt_file_paths[i], pattern2, start=1)
        else:
            text_file = __load_data(txt_file_paths[i], pattern1)

        with open("data/data_csv/" + txt_file_paths[i][14:-3] + "csv", "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow(["timestamp", "id", "add", "dlc", "data"])

            for j in range(len(text_file)):
                if text_file[j].data is None:
                    data = ""
                else:
                    data = " ".join(text_file[j].data)

                row = [text_file[j].timestamp, text_file[j].id, text_file[j].add, text_file[j].dlc, data]

                csv_writer.writerow(row)


if __name__ == "__main__":
    txt_to_csv()
