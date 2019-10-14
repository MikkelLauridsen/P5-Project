from unittest import TestCase

import message
from message import Message


class TestMessage(TestCase):
    def setUp(self) -> None:
        self.message_row = [0.000224, 809, 0, 8, "07 a7 7f 8c 11 2f 00 10"]

        self.message_object = Message(0.000224, 809, 0, 8, "07 a7 7f 8c 11 2f 00 10")

    def test_parse_csv_row(self):
        result = message.parse_csv_row(self.message_row)

        self.assertEqual(result, self.message_object)
