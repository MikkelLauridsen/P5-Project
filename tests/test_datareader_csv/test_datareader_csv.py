from unittest import TestCase
from datareader_csv import load_messages, load_datapoints
from message import Message
from datapoint import DataPoint, datapoint_attributes


class TestDatareaderCSV(TestCase):
    def setUp(self) -> None:
        time_ms = 1881258.816042923
        class_label = "normal"
        mean_id_interval = 0.011034680628272332
        variance_id_frequency = 17.72437499999999
        mean_id_intervals_variance = 8.354867823557838e-05
        mean_data_bit_count = 14.913419913419913
        variance_data_bit_count = 74.46869436479822
        kurtosis_id_interval = 74.39655141184994
        kurtosis_id_frequency = 1.323601387295182
        skewness_id_frequency = 0.5522522478213545
        kurtosis_variance_data_bit_count_id = 15.457159141707956
        skewness_id_interval_variances = 0.694951701467612

        self.message_expected = Message(0.000224, 809, 0, 8, bytearray(b'\x07\xa7\x7f\x8c\x11/\x00\x10'))

        self.datapoint_expected = DataPoint(time_ms, class_label, mean_id_interval, variance_id_frequency,
                                            mean_id_intervals_variance, mean_data_bit_count, variance_data_bit_count,
                                            kurtosis_id_interval, kurtosis_id_frequency, skewness_id_frequency,
                                            kurtosis_variance_data_bit_count_id, skewness_id_interval_variances)

    def test_load_messages(self):
        message_actual = load_messages("message.csv")[0]

        # Using almost equal to avoid the problems with floating point numbers.
        self.assertAlmostEqual(self.message_expected.timestamp, message_actual.timestamp)

        self.assertEqual(self.message_expected.id, message_actual.id)
        self.assertEqual(self.message_expected.rtr, message_actual.rtr)
        self.assertEqual(self.message_expected.dlc, message_actual.dlc)
        self.assertEqual(self.message_expected.data, message_actual.data)
