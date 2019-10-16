import math
from unittest import TestCase
import id_based_datasets
from message import Message


class TestIdBasedDatasets(TestCase):
    def setUp(self) -> None:
        # Assume
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_3 = Message(4.000000, 128, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00\x00\x53'))
        message_4 = Message(6.000000, 129, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        message_5 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        message_6 = Message(10.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        message_7 = Message(12.000000, 129, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        message_8 = Message(14.000000, 128, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        message_9 = Message(16.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        self.messages = [message_1, message_2, message_3, message_4, message_5, message_6, message_7, message_8, message_9]

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
        variance_1 = (1 / n_1) * (id_timestamp_intervals.get(128)[0] - mean_1)**2 + (1 / n_1) * (id_timestamp_intervals.get(128)[1] - mean_1)**2

        n_2 = len(id_timestamp_intervals.get(129))
        mean_2 = (id_timestamp_intervals.get(129)[0] + id_timestamp_intervals.get(129)[1]) / n_2
        variance_2 = (1 / n_2) * (id_timestamp_intervals.get(129)[0] - mean_2)**2 + (1 / n_2) * (id_timestamp_intervals.get(129)[1] - mean_2)**2

        n_3 = len(id_timestamp_intervals.get(130))
        mean_3 = (id_timestamp_intervals.get(130)[0] + id_timestamp_intervals.get(130)[1]) / n_3
        variance_3 = (1 / n_3) * (id_timestamp_intervals.get(130)[0] - mean_3)**2 + (1 / n_3) * (id_timestamp_intervals.get(130)[1] - mean_3)**2

        self.values = [variance_1, variance_2, variance_3]
        self.values.sort()
        self.median = self.values[math.floor(len(self.values) / 2)]

    def test_calculate_num_ids(self):

        # Assume
        actual_ids_seen = set()

        for message in self.messages:
            actual_ids_seen.add(message.id)

        # Action
        expected_result = id_based_datasets.calculate_num_ids(self.messages)

        # Assert
        self.assertEqual(expected_result, len(actual_ids_seen))

    def test_concat_messages(self):

        # Assume
        message_1 = Message(12.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        messages_1 = [message_1]

        message_2 = Message(4.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_3 = Message(6.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_4 = Message(8.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        messages_2 = [message_2, message_3, message_4]

        # Duplicate used to concatenate
        message_2_duplicate = Message(4.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_3_duplicate = Message(6.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_4_duplicate = Message(8.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        messages_2_duplicate = [message_2_duplicate, message_3_duplicate, message_4_duplicate]

        # Offset is calculated as
        offset = messages_1[len(messages_1) - 1].timestamp  # which is the first and only element, with timestamp 12.0

        # Messages in messages_2 will be offset with the timestamp in message_1. Thus, 4 + 12, 6 + 12, 8 + 12.
        actual_offset_message_2 = messages_2[0].timestamp + offset
        actual_offset_message_3 = messages_2[1].timestamp + offset
        actual_offset_message_4 = messages_2[2].timestamp + offset

        # Concatenate the messages in messages_1 onto messages_2_duplicate (which is the same as messages_2)
        for msgs in messages_2_duplicate:
            msgs.timestamp += offset
        actual_concat_messages_result = messages_1 + messages_2_duplicate

        # Action
        expected_result = id_based_datasets.concat_messages(messages_1, messages_2)

        # Assert
        self.assertEqual(expected_result, actual_concat_messages_result)
        self.assertEqual(expected_result[1].timestamp, actual_offset_message_2)
        self.assertEqual(expected_result[2].timestamp, actual_offset_message_3)
        self.assertEqual(expected_result[3].timestamp, actual_offset_message_4)

    def test_calculate_skewness_id_interval_variances(self):

        # Assumption when skewness can be calculated for a set of messages
        n_final = len(self.values)
        mean_final = (self.values[0] + self.values[1] + self.values[2]) / n_final
        variance_final = (1 / n_final) * (self.values[0] - mean_final) ** 2 + (1 / n_final) * (self.values[1] - mean_final) ** 2 + (1 / n_final) * (self.values[2] - mean_final) ** 2
        actual_skewness_result = (3 * (mean_final - self.median)) / math.sqrt(variance_final)

        # Assumption when skewness can NOT be calculated when unique ids is required to appear at least 3 times
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_3 = Message(4.000000, 128, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00\x00\x53'))
        message_4 = Message(6.000000, 129, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        message_5 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        message_6 = Message(10.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        no_skewness_messages = [message_1, message_2, message_3, message_4, message_5, message_6]

        # Action
        expected_result = id_based_datasets.calculate_skewness_id_interval_variances(self.messages)
        expected_no_skewness_result = id_based_datasets.calculate_skewness_id_interval_variances(no_skewness_messages)

        # Assert
        self.assertEqual(expected_result, actual_skewness_result)
        self.assertEqual(expected_no_skewness_result, 0)

    def test_calculate_mean_id_intervals_variance(self):

        # Assume
        actual_result = math.fsum(self.values) / len(self.values)

        # If length of interval variances is 0, we can't calculate anything
        # Assumption when skewness can NOT be calculated when unique ids is required to appear at least 3 times
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_3 = Message(4.000000, 128, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00\x00\x53'))
        message_4 = Message(6.000000, 129, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        message_5 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        message_6 = Message(10.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        no_mean_messages = [message_1, message_2, message_3, message_4, message_5, message_6]

        # Action
        expected_result = id_based_datasets.calculate_mean_id_intervals_variance(self.messages)
        expected_no_mean_result = id_based_datasets.calculate_skewness_id_interval_variances(no_mean_messages)

        # Assert
        self.assertEqual(expected_result, actual_result)
        self.assertEqual(expected_no_mean_result, 0)

    def test_calculate_variance_data_bit_count(self):

        # Assume
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x00\x00\x01\x01\x00\x00\x00'))
        message_3 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x00\x11\x01\x01\x01\x01\x01'))
        messages = [message_1, message_2, message_3]

        actual_counts = []
        for message in messages:
            count = 0
            for byte in message.data:
                for i in range(8):
                    if byte & (0b10000000 >> i):
                        count += 1
            actual_counts.append(count)

        actual_message_1_bit_amount = actual_counts[0]
        actual_message_2_bit_amount = actual_counts[1]
        actual_message_3_bit_amount = actual_counts[2]

        n = len(actual_counts)
        mean = (actual_counts[0] + actual_counts[1] + actual_counts[2]) / n
        result_bit_count_variance = (1 / n) * (actual_counts[0] - mean)**2 + (1 / n) * (actual_counts[1] - mean)**2 + \
                                    (1 / n) * (actual_counts[2] - mean)**2

        # If the DLC-value is 0, it must return 0.
        no_bit_count_message_1 = Message(0.000000, 128, 0, 0, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        no_bit_count_messages = [no_bit_count_message_1]

        # Action
        expected_result = id_based_datasets.calculate_variance_data_bit_count(messages)
        expected_no_bit_count = id_based_datasets.calculate_variance_data_bit_count(no_bit_count_messages)
        expected_message_1_bit_amount_result = 8
        expected_message_2_bit_amount_result = 2
        expected_message_3_bit_amount_result = 7

        # Assert
        self.assertAlmostEqual(expected_result, result_bit_count_variance, places=10)
        self.assertEqual(expected_no_bit_count, 0)
        self.assertEqual(expected_message_1_bit_amount_result, actual_message_1_bit_amount)
        self.assertEqual(expected_message_2_bit_amount_result, actual_message_2_bit_amount)
        self.assertEqual(expected_message_3_bit_amount_result, actual_message_3_bit_amount)
        #Make tests to check if message 1, 2 and 3 does gain certain amount of bits.

    def test_calculate_mean_variance_data_bit_count_id(self):

        # Assume we can calculte the mean variance data bit count
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x00\x00\x01\x01\x00\x00\x00'))
        message_3 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x00\x11\x01\x01\x01\x01\x01'))
        message_4 = Message(0.000000, 128, 0, 8, bytearray(b'\x00\x00\x00\x01\x01\x01\x01\x01'))
        message_5 = Message(2.000000, 129, 0, 8, bytearray(b'\x01\x01\x00\x00\x01\x00\x01\x01'))
        message_6 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x00\x11\x01\x00\x00\x00\x00'))
        messages = [message_1, message_2, message_3, message_4, message_5, message_6]

        id_counts = {}
        for message in messages:
            if message.id in id_counts:
                count = 0
                for byte in message.data:
                    for i in range(8):
                        if byte & (0b10000000 >> i):
                            count += 1
                id_counts[message.id].append(count)
            else:
                count = 0
                for byte in message.data:
                    for i in range(8):
                        if byte & (0b10000000 >> i):
                            count += 1
                id_counts[message.id] = [count]

        n_1 = len(id_counts.get(128))
        mean_1 = (id_counts.get(128)[0] + id_counts.get(128)[1]) / n_1
        variance_1 = (1 / n_1) * (id_counts.get(128)[0] - mean_1) ** 2 + (1 / n_1) * (
                    id_counts.get(128)[1] - mean_1) ** 2

        n_2 = len(id_counts.get(129))
        mean_2 = (id_counts.get(129)[0] + id_counts.get(129)[1]) / n_2
        variance_2 = (1 / n_2) * (id_counts.get(129)[0] - mean_2) ** 2 + (1 / n_2) * (
                    id_counts.get(129)[1] - mean_2) ** 2

        n_3 = len(id_counts.get(130))
        mean_3 = (id_counts.get(130)[0] + id_counts.get(130)[1]) / n_3
        variance_3 = (1 / n_3) * (id_counts.get(130)[0] - mean_3) ** 2 + (1 / n_3) * (
                    id_counts.get(130)[1] - mean_3) ** 2

        variance_values = [variance_1, variance_2, variance_3]
        actual_result = math.fsum(variance_values) / len(variance_values)

        # Action
        expected_result = id_based_datasets.calculate_mean_variance_data_bit_count_id(messages)
        expected_message_1_bit_amount_result = 8
        expected_message_2_bit_amount_result = 2
        expected_message_3_bit_amount_result = 7
        expected_message_4_bit_amount_result = 5
        expected_message_5_bit_amount_result = 5
        expected_message_6_bit_amount_result = 3

        # Assert
        self.assertAlmostEqual(expected_result, actual_result)
        self.assertEqual(expected_message_1_bit_amount_result, id_counts.get(128)[0])
        self.assertEqual(expected_message_2_bit_amount_result, id_counts.get(129)[0])
        self.assertEqual(expected_message_3_bit_amount_result, id_counts.get(130)[0])
        self.assertEqual(expected_message_4_bit_amount_result, id_counts.get(128)[1])
        self.assertEqual(expected_message_5_bit_amount_result, id_counts.get(129)[1])
        self.assertEqual(expected_message_6_bit_amount_result, id_counts.get(130)[1])

    def test_calculate_mean_id_interval(self):
        # Assume
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_3 = Message(4.000000, 128, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00\x00\x53'))
        messages = [message_1, message_2, message_3]

        intervals = []
        last_seen_timestamps = {}

        for message in messages:
            if message.id in last_seen_timestamps:
                intervals.append(message.timestamp - last_seen_timestamps[message.id])
            last_seen_timestamps[message.id] = message.timestamp
        actual_result = math.fsum(intervals) / len(intervals)

        # If no message with same id is present, the mean id interval CAN'T be calculated
        message_4 = Message(0.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_5 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_6 = Message(4.000000, 130, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00\x00\x53'))
        no_mean_id_interval_messages = [message_4, message_5, message_6]

        # Action
        expected_result = id_based_datasets.calculate_mean_id_interval(messages)
        expected_no_mean_id_interval_messages = id_based_datasets.calculate_mean_id_interval(no_mean_id_interval_messages)

        # Assert
        self.assertEqual(expected_result, actual_result)
        self.assertEqual(expected_no_mean_id_interval_messages, 0)

"""
    def test_calculate_kurtosis_variance_data_bit_count_id(self):

        # Action
        result = id_based_datasets.calculate_kurtosis_variance_data_bit_count_id(self.messages)

        # Assert
        self.assertAlmostEqual(result, 1.49, places=0)

    def test_calculate_mean_data_bit_count(self):

        # Action
        result = id_based_datasets.calculate_mean_data_bit_count(self.messages)

        # Assert
        self.assertEqual(result, 20.5)

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