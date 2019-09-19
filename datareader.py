import re
from recordclass import dataobject
import struct

class Message(dataobject):
    timestamp: float
    id: int
    add: int
    dlc: int
    data: bytearray

    def __str__(self):
        return f"{{{self.timestamp}, {self.id}, {self.add}, {self.dlc}, {self.data}}}"

def parse_message(message_str, pattern):
    m = re.match(pattern, message_str)
    timestamp = float(m.group("timestamp"))
    id = int(m.group("id"), 16)
    dlc = int(m.group("dlc"))

    add = 0
    try:
        add = int(m.group("add"), 2)
    except IndexError:
        pass

    data = None
    if (dlc > 0):
        raw_data = m.group("data").split(" ")
        while len(raw_data) > dlc:
            raw_data.pop()
        data = bytearray([int(i, 16) for i in raw_data])

    return Message(timestamp, id, add, dlc, data)

def load_data(filepath, pattern, start = 0, limit = None):
    data = []
    with open(filepath, "r") as f:
        for i in range(start):
            f.readline()

        c = 0
        line = f.readline()
        while line != "" and (limit == None or c < limit):
            data.append(parse_message(line, pattern))
            line = f.readline()
            c += 1
            if (c%50000==0):
                print(c)
        print(c)
    return data

pattern1 = r"Timestamp:( )*(?P<timestamp>.*)        ID: (?P<id>[0-9a-f]*)    (?P<add>[01]*)    DLC: (?P<dlc>[0-8])(    (?P<data>(([0-9a-f]+)( )?)*))?"
pattern2 = r"(?P<id>[0-9a-f]*)	(?P<dlc>[0-8])	(?P<data>(([0-9a-f]*)( )?)*)		( )*(?P<timestamp>.*)"

#Loads data from "Attack_free_dataset.txt"
def load_attack_free1(start = 0, limit = None):
    return load_data("data/Attack_free_dataset.txt", pattern1, start, limit)

#Loads data from "Attack_free_dataset2.txt"
def load_attack_free2(start = 0, limit = None):
    return load_data("data/Attack_free_dataset2.txt", pattern2, start + 1, limit)

#Loads data from "Impersonation_attack_dataset.txt"
def load_impersonation_1(start = 0, limit = None):
    return load_data("data/Impersonation_attack_dataset.txt", start, limit)

#Loads data from "170907_impersonation.txt"
def load_impersonation_2(start = 0, limit = None):
    return load_data("data/170907_impersonation.txt", start, limit)

#Loads data from "170907_impersonation_2.txt"
def load_impersonation_3(start = 0, limit = None):
    return load_data("data/170907_impersonation_2.txt", start, limit)

#Loads data from "DoS_attack_dataset.txt"
def load_dos(start = 0, limit = None):
    return load_data("data/DoS_attack_dataset.txt", start, limit)

#Loads data from "Fuzzy_attack_dataset.txt"
def load_fuzzy(start = 0, limit = None):
    return load_data("data/Fuzzy_attack_dataset.txt", start, limit)