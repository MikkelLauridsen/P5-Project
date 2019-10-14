from unittest import TestCase
from message import Message
import id_based_datasets
from id_based_datasets import calculate_skewness_id_interval_variances


class TestMessage(TestCase):
    # Assume
    message_1 = Message(0.000224, 809, 0, 8, bytearray(b'\x07\xa7\x7f\x8c\x11/\x00\x10'))
    message_2 = Message(0.000462, 128, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a/\x20\x43'))
    message_3 = Message(0.000704, 129, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00/\x00\x53'))
    message_4 = Message(0.001115, 339, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff/\x40\xce'))
    message_5 = Message(0.001366, 399, 0, 8, bytearray(b'00\x29\x20\x00\x00\x45/\x00\x00'))
    messages = [message_1, message_2, message_3, message_4, message_5]
    def test_calculate_skewness_id_interval_variances(self):


        # Action
        result1 = id_based_datasets.calculate_skewness_id_interval_variances(self.messages)

        # Assert
        self.assertEqual(result1, 0)

    def test_calculate_kurtosis_variance_data_bit_count_id(self):

        # Action
        result2 = id_based_datasets.calculate_kurtosis_variance_data_bit_count_id(self.messages)

        # Assert
        self.assertEqual(result2, 3)

    def test_calculate_mean_id_intervals_variance(self):

        # Action
        result3 = id_based_datasets.calculate_mean_id_intervals_variance(self.messages)

        # Assert
        self.assertEqual(result3, 0)

    def test_calculate_mean_data_bit_count(self):

        # Action
        result4 = id_based_datasets.calculate_mean_data_bit_count(self.messages)

        # Assert
        self.assertEqual(result4, 23.0)

    def test_calculate_variance_data_bit_count(self):

        # Action
        result5 = id_based_datasets.calculate_variance_data_bit_count(self.messages)

        # Assert
        self.assertEqual(result5, 20.8)

    def test_calculate_mean_variance_data_bit_count_id(self):

        # Action
        result6 = id_based_datasets.calculate_mean_variance_data_bit_count_id(self.messages)

        # Assert
        self.assertEqual(result6, 0.0)

    def test_calculate_mean_id_interval(self):

        # Action
        result7 = id_based_datasets.calculate_mean_id_interval(self.messages)

        # Assert
        self.assertEqual(result7, 0)

    def test_calculate_variance_id_frequency(self):

        # Action
        result8 = id_based_datasets.calculate_variance_id_frequency(self.messages)

        # Assert
        self.assertEqual(result8, 0.0)

    def test_calculate_num_id_transitions(self):

        # Action
        result9 = id_based_datasets.calculate_num_id_transitions(self.messages)

        # Assert
        self.assertEqual(result9, 4)

    def test_calculate_num_ids(self):

        # Action
        result10 = id_based_datasets.calculate_num_ids(self.messages)

        # Assert
        self.assertEqual(result10, 5)

    def test_calculate_req_to_res_time_variance(self):

        # Action
        result11 = id_based_datasets.calculate_req_to_res_time_variance(self.messages)

        # Assert
        self.assertEqual(result11, 0)

    def test_calculate_kurtosis_id_interval(self):

        # Action
        result12 = id_based_datasets.calculate_kurtosis_id_interval(self.messages)

        # Assert
        self.assertEqual(result12, 3)

    def test_calculate_skewness_id_frequency(self):

        # Action
        result13 = id_based_datasets.calculate_skewness_id_frequency(self.messages)

        # Assert
        self.assertEqual(result13, 0)

    def test_calculate_kurtosis_id_frequency(self):

        # Action
        result14 = id_based_datasets.calculate_kurtosis_id_frequency(self.messages)

        # Assert
        self.assertEqual(result14, 3)

    def test_calculate_kurtosis_mean_id_intervals(self):

        # Action
        result15 = id_based_datasets.calculate_kurtosis_mean_id_intervals(self.messages)

        # Assert
        self.assertEqual(result15, 3)

    def test_calculate_kurtosis_mean_id_intervals(self):

        # Action
        result16 = id_based_datasets.calculate_kurtosis_req_to_res_time(self.messages)

        # Assert
        self.assertEqual(result16, 3)
