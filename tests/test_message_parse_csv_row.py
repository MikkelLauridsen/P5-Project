from unittest import TestCase
import message
from message import Message
from message import parse_csv_row

class TestMessage_parse_csv_row(TestCase):

    def test_parse_csv_row(self):
        # Assume
        message_test = [0.000224, 809, 0, 8, "07 a7 7f 8c 11 2f 00 10"]

        # Action
        result = message.parse_csv_row(message_test)

        # Assert
        self.assertEqual(result.timestamp, 0.000224)
        self.assertEqual(result.id, 809)
        self.assertEqual(result.add, 0)
        self.assertEqual(result.dlc, 8)
        self.assertEqual(result.data, bytearray(b'\x07\xa7\x7f\x8c\x11/\x00\x10'))
        self.assertTrue(result, Message)
