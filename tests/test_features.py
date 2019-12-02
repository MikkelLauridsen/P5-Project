import features
from unittest import TestCase
from message import Message


class TestFeatures(TestCase):
    def setUp(self) -> None:
        self.messages = [
            Message(0.000224, 809, 0, 8, bytearray(b'\x07\xa7\x7f\x8c\x11/\x00\x10')),
            Message(0.000230, 709, 0, 8, bytearray(b'\x00\x00\x00\x00\x00\00\x00\x00')),
            Message(0.000232, 809, 0, 4, bytearray(b'\x00\x00\x00\x00')),
            Message(0.000233, 709, 0, 4, bytearray(b'\x00\x00\x00\x00')),
            Message(0.000235, 809, 0, 6, bytearray(b'\xff\x00\xff\x00\x00\00'))]

    def test_calculate_skewness_id_interval_variances(self):
        skewness = features.calculate_skewness_id_interval_variances(self.messages)
        expected = -3.0000000000000004  # Per Pearson's second skewness coefficient

        self.assertAlmostEqual(skewness, expected)

    def test_calculate_kurtosis_variance_data_bit_count_id(self):
        kurtosis = features.calculate_kurtosis_variance_data_bit_count_id(self.messages)
        expected = 1.0  # Per Bock's kurtosis coefficient

        self.assertAlmostEqual(kurtosis, expected)

    def test_calculate_mean_id_intervals_variance(self):
        mean = features.calculate_mean_id_intervals_variance(self.messages)
        expected = 0.000000000003125

        self.assertAlmostEqual(mean, expected)

    def test_calculate_mean_data_bit_count(self):
        mean = features.calculate_mean_data_bit_count(self.messages)
        expected = 8.4

        self.assertAlmostEqual(mean, expected)

    def test_calculate_variance_data_bit_count(self):
        variance = features.calculate_variance_data_bit_count(self.messages)
        expected = 115.84

        self.assertAlmostEqual(variance, expected)

    def test_calculate_mean_variance_data_bit_count_id(self):
        mean = features.calculate_mean_variance_data_bit_count_id(self.messages)
        expected = 57.333333333333336

        self.assertAlmostEqual(mean, expected)


