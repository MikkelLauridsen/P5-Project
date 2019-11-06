"""
The Message class describing an object containing a single CAN bus message.

Functions:
parse_csv_row: Parses a row from a csv file into a corresponding message object.
get_csv_row: Returns a list containing every attribute in the given Message object.
"""
from recordclass import dataobject


class Message(dataobject):
    """Class representing a single CAN bus message."""
    timestamp: float
    id: int
    rtr: int
    dlc: int
    data: bytearray

    def __str__(self):
        return f"{{{self.timestamp}, {self.id}, {self.rtr}, {self.dlc}, {self.data}}}"


message_attributes = Message.__annotations__.keys()


def parse_csv_row(row):
    """Parses a single row from a csv file and returns the corresponding Message object."""
    timestamp = float(row[0])
    id = int(row[1])
    rtr = int(row[2])
    dlc = int(row[3])

    data = None
    if dlc > 0:
        raw_data = row[4].split(" ")  # Split data-field into list with data-pairs.
        while len(raw_data) > dlc:  # If data-field is larger than DLC, pop until the same size.
            raw_data.pop()
        data = bytearray([int(i, 16) for i in raw_data])

    return Message(timestamp, id, rtr, dlc, data)


def get_csv_row(message):
    """Creates a list of the given messages components and returns it to facilitate csv file creation."""
    row = [
        "%.6f" % message.timestamp,
        message.id,
        message.rtr,
        message.dlc,
        ""
    ]

    br = message.data
    if br is not None:
        row[4] = " ".join(["%02x" % b for b in br])

    return row
