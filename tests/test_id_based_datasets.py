import math
from unittest import TestCase

import id_based_datasets
from message import Message


class TestIdBasedDatasets(TestCase):
    def setUp(self) -> None:
        # Assume
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f/\x00\x10'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a/\x20\x43'))
        message_3 = Message(4.000000, 128, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00/\x00\x53'))
        message_4 = Message(6.000000, 129, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff/\x40\xce'))
        message_5 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45/\x00\x00'))
        message_6 = Message(10.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45/\x00\x00'))
        message_7 = Message(12.000000, 129, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45/\x00\x00'))
        message_8 = Message(14.000000, 128, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45/\x00\x00'))
        message_9 = Message(16.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45/\x00\x00'))
        self.messages = [message_1, message_2, message_3, message_4, message_5, message_6, message_7, message_8, message_9]

    def test_calculate_num_ids(self):
        # Action
        result = id_based_datasets.calculate_num_ids(self.messages)
        ids_seen = set()

        for message in self.messages:
            ids_seen.add(message.id)

        # Assert
        self.assertEqual(result, len(ids_seen))

    def concat_messages(self):

        # Assume
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f/\x00\x10'))
        messages_1 = [message_1]

        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a/\x20\x43'))
        messages_2 = [message_2]

        concat_messages = [message_1, message_2]

        # Action
        result = id_based_datasets.concat_messages(messages_1, messages_2)

        # Assert
        self.assertEqual(result, concat_messages)

    def test_calculate_skewness_id_interval_variances(self):

        # Find interval between same ids.
        id_timestamp_intervals = {}
        last_seen_timestamps = {}

        for message in self.messages:
            if message.id in last_seen_timestamps:
                interval = message.timestamp - last_seen_timestamps[message.id]
                id_timestamp_intervals.setdefault(message.id, [])
                id_timestamp_intervals[message.id].append(interval)

            last_seen_timestamps[message.id] = message.timestamp
        n_1 = len(id_timestamp_intervals.get(128))
        mean_1 = (id_timestamp_intervals.get(128)[0] + id_timestamp_intervals.get(128)[1]) / n_1
        variance_1 = (1 / n_1) * (id_timestamp_intervals.get(128)[0] - mean_1) ** 2 + (1 / n_1) * (id_timestamp_intervals.get(128)[1] - mean_1) ** 2

        n_2 = len(id_timestamp_intervals.get(129))
        mean_2 = (id_timestamp_intervals.get(129)[0] + id_timestamp_intervals.get(129)[1]) / n_2
        variance_2 = (1 / n_2) * (id_timestamp_intervals.get(129)[0] - mean_2) ** 2 + (1 / n_2) * (id_timestamp_intervals.get(129)[1] - mean_2) ** 2

        n_3 = len(id_timestamp_intervals.get(130))
        mean_3 = (id_timestamp_intervals.get(130)[0] + id_timestamp_intervals.get(130)[1]) / n_3
        variance_3 = (1 / n_3) * (id_timestamp_intervals.get(130)[0] - mean_3) ** 2 + (1 / n_3) * (id_timestamp_intervals.get(130)[1] - mean_3) ** 2

        values = [variance_1, variance_2, variance_3]
        values.sort()
        median = values[math.floor(len(values) / 2)]

        n_final = len(values)
        mean_final = (values[0] + values[1] + values[2]) / n_final
        variance_final = (1 / n_final) * (values[0] - mean_final) ** 2 + (1 / n_final) * (values[1] - mean_final) ** 2 + (1 / n_final) * (values[2] - mean_final) ** 2
        skewness_final = (3 * (mean_final - median)) / math.sqrt(variance_final)

        # Action
        result = id_based_datasets.calculate_skewness_id_interval_variances(self.messages)


        # Assert
        self.assertEqual(result, skewness_final)
"""
    def test_calculate_kurtosis_variance_data_bit_count_id(self):

        # Action
        result = id_based_datasets.calculate_kurtosis_variance_data_bit_count_id(self.messages)

        # Assert
        self.assertAlmostEqual(result, 1.49, places=0)

    def test_calculate_mean_id_intervals_variance(self):

        # Action
        result = id_based_datasets.calculate_mean_id_intervals_variance(self.messages)

        # Assert
        self.assertEqual(result, 0)

    def test_calculate_mean_data_bit_count(self):

        # Action
        result = id_based_datasets.calculate_mean_data_bit_count(self.messages)

        # Assert
        self.assertEqual(result, 20.5)

    def test_calculate_variance_data_bit_count(self):

        # Action
        result = id_based_datasets.calculate_variance_data_bit_count(self.messages)

        # Assert
        self.assertEqual(result, 43.25)

    def test_calculate_mean_variance_data_bit_count_id(self):

        # Action
        result = id_based_datasets.calculate_mean_variance_data_bit_count_id(self.messages)

        # Assert
        self.assertAlmostEqual(result, 5.08, places=0)

    def test_calculate_mean_id_interval(self):
        # Action
        result = id_based_datasets.calculate_mean_id_interval(self.messages)

        # Assert
        self.assertAlmostEqual(result, 3.33, places=0)

    def test_calculate_variance_id_frequency(self):
        # Action
        result = id_based_datasets.calculate_variance_id_frequency(self.messages)

        # Assert
        self.assertEqual(result, 0)

    def test_calculate_num_id_transitions(self):
        # Action
        result = id_based_datasets.calculate_num_id_transitions(self.messages)

        # Assert
        self.assertEqual(result, 4)

    def test_calculate_req_to_res_time_variance(self):
        # Action
        result = id_based_datasets.calculate_req_to_res_time_variance(self.messages)

        # Assert
        self.assertEqual(result, 0)

    def test_calculate_kurtosis_id_interval(self):
        # Action
        result = id_based_datasets.calculate_kurtosis_id_interval(self.messages)

        # Assert
        self.assertAlmostEqual(result, 1.5, places=0)

    def test_calculate_skewness_id_frequency(self):
        # Action
        result = id_based_datasets.calculate_skewness_id_frequency(self.messages)

        # Assert
        self.assertEqual(result, 0)

    def test_calculate_kurtosis_id_frequency(self):
        # Action
        result = id_based_datasets.calculate_kurtosis_id_frequency(self.messages)

        # Assert
        self.assertEqual(result, 3)

    def test_calculate_kurtosis_mean_id_intervals(self):
        # Action
        result = id_based_datasets.calculate_kurtosis_mean_id_intervals(self.messages)

        # Assert
        self.assertAlmostEqual(result, 1.5, places=0)

    def test_calculate_kurtosis_req_to_res_time(self):
        # Action
        result = id_based_datasets.calculate_kurtosis_req_to_res_time(self.messages)

        # Assert
        self.assertAlmostEqual(result, 3, places=0)
"""