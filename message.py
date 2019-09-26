from recordclass import dataobject


class Message(dataobject):
    timestamp: float
    id: int
    add: int
    dlc: int
    data: bytearray

    def __str__(self):
        return f"{{{self.timestamp}, {self.id}, {self.add}, {self.dlc}, {self.data}}}"


def parse_csv_row(row):
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