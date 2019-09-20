import csv
import datareader_old
import os

# Setting up a link between each txt file function and their equivalent csv file.
txt_to_csv = [(datareader_old.load_impersonation_2, "data/170907_impersonation.csv"),
              (datareader_old.load_impersonation_3, "data/170907_impersonation_2.csv"),
              (datareader_old.load_attack_free1, "data/Attack_free_dataset.csv"),
              (datareader_old.load_attack_free2, "data/Attack_free_dataset2.csv"),
              (datareader_old.load_dos, "data/DoS_attack_dataset.csv"),
              (datareader_old.load_fuzzy, "data/Fuzzy_attack_dataset.csv"),
              (datareader_old.load_impersonation_1, "data/Impersonation_attack_dataset.csv")]

# Going through each link between txt file functions and csv files.
for i in range(len(txt_to_csv)):
    text_file = txt_to_csv[i][0](mode="txt_to_csv")

    with open(txt_to_csv[i][1], "w", newline="") as csv_file:
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
        os.remove(txt_to_csv[i][1][:-3] + "txt")
