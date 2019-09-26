import csv
import message
import idpoint


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
    return __load_data(filepath, message.parse_csv_row, start, limit)


def load_idpoints(filepath, start = 0, limit = None):
    return __load_data(filepath, idpoint.parse_csv_row, start, limit)


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
