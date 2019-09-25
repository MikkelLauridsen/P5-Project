from recordclass import dataobject


class Message(dataobject):
    timestamp: float
    id: int
    add: int
    dlc: int
    data: bytearray

    def __str__(self):
        return f"{{{self.timestamp}, {self.id}, {self.add}, {self.dlc}, {self.data}}}"