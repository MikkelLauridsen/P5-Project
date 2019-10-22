"""Functions for manipulating data."""

import pandas as pd
import random
import os


def manipulate_dos(filepath, target_dir):
    """Replaces DoS injection messages with more realistic attacks and outputs the changed file in the target dir."""
    # Creating the target directory if it does not exist.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Getting the data from the filepath.
    data = pd.read_csv(filepath)

    # Getting the "normal" data fields, hereby meaning non-attack messages that are not remote frames.
    normal_data_fields = __get_normal_data_fields(data)

    data = __replace_dos_messages(data, normal_data_fields)

    # Converting the manipulated dataset back into a csv file.
    data.to_csv(f"{target_dir}DoS_manipulated.csv", columns=["timestamp", "id", "rtr", "dlc", "data"], index=False)


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


if __name__ == "__main__":
    manipulate_dos("data/data_csv/DoS_attack_dataset.csv", "data/manipulated_data/")
