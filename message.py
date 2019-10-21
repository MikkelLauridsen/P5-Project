from recordclass import dataobject


# Class representing the messages in the .csv file(s)
class Message(dataobject):
    timestamp: float
    id: int
    rtr: int
    dlc: int
    data: bytearray

    def __str__(self):
        return f"{{{self.timestamp}, {self.id}, {self.rtr}, {self.dlc}, {self.data}}}"


# Function to read single line FROM .csv file(s) and return a Message dataobject.
def parse_csv_row(row):
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
