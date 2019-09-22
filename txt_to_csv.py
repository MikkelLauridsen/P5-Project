import csv
import datareader_txt
import os

# Creating the directory if it does not exist
if not os.path.exists("data_csv"):
    os.makedirs("data_csv")

# Getting a list of links between each txt file function and their equivalent csv file.
load_functions_and_paths = datareader_txt.get_load_functions_and_paths()

# Going through each link between txt file functions and csv files.
for i in range(len(load_functions_and_paths)):
    text_file = load_functions_and_paths[i][0](mode="txt_to_csv")

    with open(load_functions_and_paths[i][1][:-3] + "csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(["timestamp", "id", "add", "dlc", "data"])

        for j in range(len(text_file)):
            if text_file[j].data is None:
                data = ""
            else:
                data = " ".join(text_file[j].data)

            row = [text_file[j].timestamp, text_file[j].id, text_file[j].add, text_file[j].dlc, data]

            csv_writer.writerow(row)

        # Removing the txt file after it has been converted to a csv file.
        # os.remove(txt_to_csv[i][1][:-3] + "txt")
