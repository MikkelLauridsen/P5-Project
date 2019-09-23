import datareader_csv
import csv
from decimal import Decimal
import os
import pandas as pd


# Creating a csv containing the amount of messages within variable sized time slices.
def amount_of_messages(milliseconds, load_functions_paths, mode):
    # Converting the amount of milliseconds to its value in terms of seconds.
    seconds = milliseconds / 1000

    for i in range(len(load_functions_paths)):
        # Getting a list of every message in the current file.
        messages = load_functions_paths[i][0]()

        timestamp_counter = messages[0].timestamp + seconds
        message_counter = 0

        # Creating a csv path for the new file using the corresponding csv file currently loaded from.
        csv_path = "frequency_features_data/" + load_functions_paths[i][1][9:-4] + "_" + str(milliseconds) + ".csv"

        # Creating a list that can be used to keep track of the visited id's in unique mode.
        unique_messages = []

        with open(csv_path, "w", newline="") as datafile:
            datafile_writer = csv.writer(datafile, delimiter=",")

            # Writing the header.
            datafile_writer.writerow(["interval_" + str(milliseconds), "frequency_" + str(milliseconds)])

            for message in messages:
                # If the message timestamp is lower than the counter then we have one more message in that time slice.
                if message.timestamp <= timestamp_counter:
                    if mode == "unique":
                        if message.id not in unique_messages:
                            unique_messages.append(message.id)
                            message_counter += 1
                    else:
                        message_counter += 1
                # If not then we write the time slice and the corresponding message frequency to the csv file.
                # The last time slice is ignored since it is not a complete time slice.
                else:
                    # Using Decimal to deal with floating point math.
                    datafile_writer.writerow(["{0:.2f}".format(Decimal(str(timestamp_counter)) - Decimal(str(seconds)))
                                              + "-" + "{0:.2f}".format(timestamp_counter), message_counter])

                    timestamp_counter = float(Decimal(str(timestamp_counter)) + Decimal(str(seconds)))
                    if mode == "unique":
                        unique_messages.clear()
                        unique_messages.append(message.id)
                    message_counter = 1


# Running the amount_of_messages function multiple times and merging the answers.
def run_and_merge(mode):
    # Creating the directory if it does not exist.
    if not os.path.exists("frequency_features_data/" + mode):
        os.makedirs("frequency_features_data/" + mode)

    # Getting all the load functions and their paths so each file can be worked on in a sequential manner.
    load_functions_paths = datareader_csv.get_load_functions_and_paths()

    # Running the function three times for 10, 100 and 1000 ms respectively.
    amount_of_messages(10, load_functions_paths, mode)
    amount_of_messages(100, load_functions_paths, mode)
    amount_of_messages(1000, load_functions_paths, mode)

    # Going through the paths in the list of tuples.
    for path in load_functions_paths:
        # Extracting the file name without directory path.
        path = path[1][9:-4]

        ms_10 = pd.read_csv("frequency_features_data/" + path + "_10.csv")
        ms_100 = pd.read_csv("frequency_features_data/" + path + "_100.csv")
        ms_1000 = pd.read_csv("frequency_features_data/" + path + "_1000.csv")

        # Merging the three files horizontally.
        pd.concat([ms_10, ms_100, ms_1000], axis=1).to_csv("frequency_features_data/" + mode + "/" + path + "_" + mode +
                                                           "_message_frequency.csv", index=False, na_rep="N/A")

        # Deleting the files that are now represented by the single merged file.
        os.remove("frequency_features_data/" + path + "_10.csv")
        os.remove("frequency_features_data/" + path + "_100.csv")
        os.remove("frequency_features_data/" + path + "_1000.csv")


run_and_merge("normal")
run_and_merge("unique")
