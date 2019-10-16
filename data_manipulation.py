"""Functions for manipulating data."""

import pandas as pd
import random
import os


def manipulate_dos(filepath, target_dir):
    """Replaces DoS injection messages with more realistic attacks and outputs the changed file in the target dir."""
    # Creating the target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Getting the data from the filepath
    data = pd.read_csv(filepath)

    attack_free_data_fields = __get_attack_free_data_fields(data)
    # TODO: Maybe remove the empty data fields from attack free data fields

    # Going through all DoS injected messages and replacing the data field with a random attack free data field.
    for row in data.itertuples():
        if row.id == 0:
            # Setting the seed with "Random(row.Index)" so it is consistent across multiple program executions.
            data.at[row.Index, "data"] = random.Random(row.Index).choice(attack_free_data_fields)

    # Converting the manipulated dataset back into a csv file.
    data.to_csv(f"{target_dir}DoS_manipulated.csv", columns=["timestamp", "id", "add", "dlc", "data"], index=False)


# Goes through a dataframe and returns list containing all the data fields for non-attack messages. Only works for DoS.
def __get_attack_free_data_fields(data):
    # Getting all DoS attack free messages in the file
    attack_free_data = data[data["id"] != 0]

    return attack_free_data["data"].tolist()


if __name__ == "__main__":
    manipulate_dos("data/data_csv/DoS_attack_dataset.csv", "data/manipulated_data/")
