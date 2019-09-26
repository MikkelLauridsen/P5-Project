import csv
from message import Message
from idpoint import IDPoint


def __parse_message(row):
    timestamp = float(row[0])
    id = int(row[1])
    add = int(row[2])
    dlc = int(row[3])

    data = None
    if dlc > 0:
        raw_data = row[4].split(" ")
        while len(raw_data) > dlc:
            raw_data.pop()
        data = bytearray([int(i, 16) for i in raw_data])

    return Message(timestamp, id, add, dlc, data)


def __parse_idpoint(row):
    time_ms = float(row[0])
    is_injected = True if row[1] == "True" else False
    mean_id_interval = float(row[2])
    variance_id_frequency = float(row[3])
    num_id_transitions = int(row[4])
    num_ids = int(row[5])
    num_msgs = int(row[6])

    return IDPoint(time_ms, is_injected, mean_id_interval, variance_id_frequency, num_id_transitions, num_ids, num_msgs)


def __load_data(filepath, parse_func, start, limit):
    data = []
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)

        # Skipping the header.
        next(csv_reader)

        # Skipping however many places the caller specified.
        for i in range(start):
            next(csv_reader)

        c = 0
        for row in csv_reader:
            if limit is None or c < limit:
                data.append(parse_func(row))
                c += 1
                if c % 50000 == 0:
                    print(c)
            else:
                print(c)
                return data
        print(c)
        return data


def load_messages(filepath, start = 0, limit = None):
    return __load_data(filepath, __parse_message, start, limit)


def load_idpoints(filepath, start = 0, limit = None):
    return __load_data(filepath, __parse_idpoint, start, limit)


# Loads data from "Attack_free_dataset.csv"
def load_attack_free1(start=0, limit=None):
    return load_messages("data_csv/Attack_free_dataset.csv", start, limit)


# Loads data from "Attack_free_dataset2.csv"
def load_attack_free2(start=0, limit=None):
    return load_messages("data_csv/Attack_free_dataset2.csv", start, limit)


# Loads data from "Impersonation_attack_dataset.csv"
def load_impersonation_1(start=0, limit=None):
    return load_messages("data_csv/Impersonation_attack_dataset.csv", start, limit)


# Loads data from "170907_impersonation.csv"
def load_impersonation_2(start=0, limit=None):
    return load_messages("data_csv/170907_impersonation.csv", start, limit)


# Loads data from "170907_impersonation_2.csv"
def load_impersonation_3(start=0, limit=None):
    return load_messages("data_csv/170907_impersonation_2.csv", start, limit)


# Loads data from "DoS_attack_dataset.csv"
def load_dos(start=0, limit=None):
    return load_messages("data_csv/DoS_attack_dataset.csv", start, limit)


# Loads data from "Fuzzy_attack_dataset.csv"
def load_fuzzy(start=0, limit=None):
    return load_messages("data_csv/Fuzzy_attack_dataset.csv", start, limit)


# Returning a list containing all the different load functions and corresponding paths in no particular order.
def get_load_functions_and_paths():
    return [(load_attack_free1, "data_csv/Attack_free_dataset.csv"),
            (load_attack_free2, "data_csv/Attack_free_dataset2.csv"),
            (load_impersonation_1, "data_csv/Impersonation_attack_dataset.csv"),
            (load_impersonation_2, "data_csv/170907_impersonation.csv"),
            (load_impersonation_3, "data_csv/170907_impersonation_2.csv"),
            (load_dos, "data_csv/DoS_attack_dataset.csv"),
            (load_fuzzy, "data_csv/Fuzzy_attack_dataset.csv")]
