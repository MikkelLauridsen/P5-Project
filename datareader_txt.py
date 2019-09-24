import re

from recordclass import dataobject


class Message(dataobject):
    timestamp: float
    id: int
    add: int
    dlc: int
    data: bytearray

    def __str__(self):
        return f"{{{self.timestamp}, {self.id}, {self.add}, {self.dlc}, {self.data}}}"


def parse_message(message_str, pattern, mode):
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
    if mode == "txt":
        if dlc > 0:
            raw_data = m.group("data").split(" ")
            while len(raw_data) > dlc:
                raw_data.pop()
            raw_data = bytearray([int(i, 16) for i in raw_data])
    elif mode == "txt_to_csv":
        if dlc > 0:
            raw_data = m.group("data").split(" ")

    return Message(timestamp, id, add, dlc, raw_data)


def load_data(filepath, pattern, start=0, limit=None, mode="txt"):
    data = []
    with open(filepath, "r") as f:
        for i in range(start):
            f.readline()

        c = 0
        line = f.readline()
        while line != "" and (limit is None or c < limit):
            data.append(parse_message(line, pattern, mode))
            line = f.readline()
            c += 1
            if c % 50000 == 0:
                print(c)
        print(c)
    return data


pattern1 = r"Timestamp:( )*(?P<timestamp>.*)        ID: (?P<id>[0-9a-f]*)    (?P<add>[01]*)    "\
           r"DLC: (?P<dlc>[0-8])(    (?P<data>(([0-9a-f]+)( )?)*))?"
pattern2 = r"(?P<id>[0-9a-f]*)	(?P<dlc>[0-8])	(?P<data>(([0-9a-f]*)( )?)*)		( )*(?P<timestamp>.*)"


# Loads data from "Attack_free_dataset.txt"
def load_attack_free1(start=0, limit=None, mode="txt"):
    return load_data("data/Attack_free_dataset.txt", pattern1, start, limit, mode)


# Loads data from "Attack_free_dataset2.txt"
def load_attack_free2(start=0, limit=None, mode="txt"):
    return load_data("data/Attack_free_dataset2.txt", pattern2, start + 1, limit, mode)


# Loads data from "Impersonation_attack_dataset.txt"
def load_impersonation_1(start=0, limit=None, mode="txt"):
    return load_data("data/Impersonation_attack_dataset.txt", pattern1, start, limit, mode)


# Loads data from "170907_impersonation.txt"
def load_impersonation_2(start=0, limit=None, mode="txt"):
    return load_data("data/170907_impersonation.txt", pattern1, start, limit, mode)


# Loads data from "170907_impersonation_2.txt"
def load_impersonation_3(start=0, limit=None, mode="txt"):
    return load_data("data/170907_impersonation_2.txt", pattern1, start, limit, mode)


# Loads data from "DoS_attack_dataset.txt"
def load_dos(start=0, limit=None, mode="txt"):
    return load_data("data/DoS_attack_dataset.txt", pattern1, start, limit, mode)


# Loads data from "Fuzzy_attack_dataset.txt"
def load_fuzzy(start=0, limit=None, mode="txt"):
    return load_data("data/Fuzzy_attack_dataset.txt", pattern1, start, limit, mode)


# Returning a list containing all the different load functions and corresponding paths in no particular order.
def get_load_functions_and_paths():
    return [(load_attack_free1, "data_csv/Attack_free_dataset.txt"),
            (load_attack_free2, "data_csv/Attack_free_dataset2.txt"),
            (load_impersonation_1, "data_csv/Impersonation_attack_dataset.txt"),
            (load_impersonation_2, "data_csv/170907_impersonation.txt"),
            (load_impersonation_3, "data_csv/170907_impersonation_2.txt"),
            (load_dos, "data_csv/DoS_attack_dataset.txt"),
            (load_fuzzy, "data_csv/Fuzzy_attack_dataset.txt")]