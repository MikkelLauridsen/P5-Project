import re
import os
import csv

import message


# This function uses regular expression to match
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


def txt_to_csv(start_dir, target_dir):
    """Taking all txt files in the start dir, converting them to csv files and putting them in the target dir."""
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

        # Creating a corresponding csv file and opening it in write mode.
        with open(f"{path.replace(start_dir, target_dir)[:-3]}" + "csv", "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # Adding a header to the csv file.
            csv_writer.writerow(["timestamp", "id", "add", "dlc", "data"])

            # Adding all the txt file data to the csv file.
            for j in range(len(text_file)):
                if text_file[j].data is None:
                    data = ""
                else:
                    data = " ".join(text_file[j].data)

                row = [text_file[j].timestamp, text_file[j].id, text_file[j].add, text_file[j].dlc, data]

                csv_writer.writerow(row)


if __name__ == "__main__":
    txt_to_csv("data/raw_data/", "data/data_csv/")
