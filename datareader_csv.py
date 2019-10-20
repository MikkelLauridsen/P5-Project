import pandas as pd

import idpoint
import message


def __load_data(filepath, parse_func, start, limit):
    print(f"Started reading from {filepath}")

    data = []

    try:
        # Creating a dataframe that contains the specified rows of the specified csv file.
        df = pd.read_csv(filepath, header=0, skiprows=range(1, start + 1), nrows=limit)
    except FileNotFoundError:
        print("The file does not exist")
        return

    matching_header, diff = idpoint.is_header_matching(df.columns)
    if not matching_header:
        print("Found mismatching datapoint and csv file structure")
        print(f"idpoint: {list(idpoint.idpoint_attributes)}")
        print(f"csv    : {list(df.columns)}")
        print(f"diff   : {diff}")

    for count, row in enumerate(df.values.tolist()):
        if count % 50000 == 0:
            print(f"Reading message: {str(count)}")
        data.append(parse_func(row))

    print("Completed")
    return data


def load_messages(filepath, start=0, limit=None):
    return __load_data(filepath, message.parse_csv_row, start, limit)


def load_idpoints(filepath, start=0, limit=None):
    return __load_data(filepath, idpoint.parse_csv_row, start, limit)


# Loads data from "Attack_free_dataset.csv"
def load_attack_free1(start=0, limit=None):
    return load_messages("data/data_csv/Attack_free_dataset.csv", start, limit)


# Loads data from "Attack_free_dataset2.csv"
def load_attack_free2(start=0, limit=None):
    return load_messages("data/data_csv/Attack_free_dataset2.csv", start, limit)


# Loads data from "Impersonation_attack_dataset.csv"
def load_impersonation_1(start=0, limit=None):
    return load_messages("data/data_csv/Impersonation_attack_dataset.csv", start, limit)


# Loads data from "170907_impersonation.csv"
def load_impersonation_2(start=0, limit=None):
    return load_messages("data/data_csv/170907_impersonation.csv", start, limit)


# Loads data from "170907_impersonation_2.csv"
def load_impersonation_3(start=0, limit=None):
    return load_messages("data/data_csv/170907_impersonation_2.csv", start, limit)


# Loads data from "DoS_attack_dataset.csv"
def load_dos(start=0, limit=None):
    return load_messages("data/data_csv/DoS_attack_dataset.csv", start, limit)


# Loads data from "Fuzzy_attack_dataset.csv"
def load_fuzzy(start=0, limit=None):
    return load_messages("data/data_csv/Fuzzy_attack_dataset.csv", start, limit)
