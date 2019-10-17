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
        self.id_timestamp_intervals = {}
        last_seen_timestamps = {}

        for message in self.messages:
            if message.id in last_seen_timestamps:
                interval = message.timestamp - last_seen_timestamps[message.id]
                self.id_timestamp_intervals.setdefault(message.id, [])
                self.id_timestamp_intervals[message.id].append(interval)

            last_seen_timestamps[message.id] = message.timestamp
        n_1 = len(self.id_timestamp_intervals.get(128))
        mean_1 = (self.id_timestamp_intervals.get(128)[0] + self.id_timestamp_intervals.get(128)[1]) / n_1
        variance_1 = (1 / n_1) * (self.id_timestamp_intervals.get(128)[0] - mean_1)**2 + (1 / n_1) * (self.id_timestamp_intervals.get(128)[1] - mean_1)**2

        n_2 = len(self.id_timestamp_intervals.get(129))
        mean_2 = (self.id_timestamp_intervals.get(129)[0] + self.id_timestamp_intervals.get(129)[1]) / n_2
        variance_2 = (1 / n_2) * (self.id_timestamp_intervals.get(129)[0] - mean_2)**2 + (1 / n_2) * (self.id_timestamp_intervals.get(129)[1] - mean_2)**2

        n_3 = len(self.id_timestamp_intervals.get(130))
        mean_3 = (self.id_timestamp_intervals.get(130)[0] + self.id_timestamp_intervals.get(130)[1]) / n_3
        variance_3 = (1 / n_3) * (self.id_timestamp_intervals.get(130)[0] - mean_3)**2 + (1 / n_3) * (self.id_timestamp_intervals.get(130)[1] - mean_3)**2

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

    def test_calculate_kurtosis_id_interval(self):

        # Assume
        intervals = []
        last_seen_timestamps = {}

        for message in self.messages:
            if message.id in last_seen_timestamps:
                intervals.append(message.timestamp - last_seen_timestamps[message.id])
            last_seen_timestamps[message.id] = message.timestamp
        n = len(intervals)
        mean = math.fsum(intervals) / len(intervals)
        variance = (1 / n) * (intervals[0] - mean)**2 + (1 / n) * (intervals[1] - mean)**2 + \
                   (1 / n) * (intervals[2] - mean)**2 + (1 / n) * (intervals[3] - mean)**2 + \
                   (1 / n) * (intervals[4] - mean)**2 + (1 / n) * (intervals[5] - mean)**2


        deviation = 0
        for elem in intervals:
            deviation += (elem - mean) ** 4
        actual_kurtosis_result = 1 / len(intervals) * deviation / variance ** 2

        # Action
        expected_kurtosis_result = id_based_datasets.calculate_kurtosis_id_interval(self.messages)

        # Assert
        self.assertAlmostEqual(expected_kurtosis_result, actual_kurtosis_result)

    def test_calculate_kurtosis_mean_id_intervals(self):

        # Assume
        interval_means = []
        for intervals in self.id_timestamp_intervals.values():
            interval_means.append(sum(intervals) / len(intervals))

        n = len(interval_means)
        mean = math.fsum(interval_means) / len(interval_means)
        variance = (1 / n) * (interval_means[0] - mean)**2 + \
                   (1 / n) * (interval_means[1] - mean)**2 + \
                   (1 / n) * (interval_means[2] - mean)**2

        deviation = 0
        for elem in interval_means:
            deviation += (elem - mean) ** 4
        actual_kurtosis__mean_result = 1 / len(interval_means) * deviation / variance ** 2

        # Action
        expected_kurtosis_mean_result = id_based_datasets.calculate_kurtosis_mean_id_intervals(self.messages)

        # Assert
        self.assertAlmostEqual(expected_kurtosis_mean_result, actual_kurtosis__mean_result)

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

    def test_calculate_mean_data_bit_count(self):

        # Assume
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
        mean = 0
        for key, value in id_counts.items():
            calc_mean = (value[0] + value[1]) / len(value)
            mean += calc_mean
        actual_mean_result = mean / len(id_counts)

        # Action
        expected_mean_result = id_based_datasets.calculate_mean_data_bit_count(messages)

        # Assert
        self.assertEqual(expected_mean_result, actual_mean_result)

    def test_calculate_skewness_id_frequency(self):

        # Assume
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x00\x00\x01\x01\x00\x00\x00'))
        message_3 = Message(4.000000, 130, 0, 8, bytearray(b'\x00\x00\x11\x01\x01\x01\x01\x01'))
        message_4 = Message(8.000000, 130, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        messages = [message_1, message_2, message_3, message_4]
        frequencies = {}

        for message in messages:
            if message.id not in frequencies:
                frequencies[message.id] = 1
            else:
                frequencies[message.id] += 1
        values = list(frequencies.values())

        n = len(values)
        mean = math.fsum(values) / n
        variance = 1 / n * (values[0] - mean) ** 2 + 1 / n * (values[1] - mean) ** 2 + 1 / n * (
                values[2] - mean) ** 2
        if variance != 0:
            values.sort()
            median = values[math.floor(len(values) / 2)]
            actual_skewness_result = (3 * (mean - median)) / math.sqrt(variance)
        else:
            actual_skewness_result = 0

        # Skewness cannot be calculated for same frequencies for all messages, due to having no variance.
        # Should result to 0.
        message_5 = Message(0.000000, 128, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        message_6 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x00\x00\x01\x01\x00\x00\x00'))
        message_7 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x00\x11\x01\x01\x01\x01\x01'))
        no_skewness_messages = [message_5, message_6, message_7]

        no_skewness_frequencies = {}

        for message in no_skewness_messages:
            if message.id not in no_skewness_frequencies:
                no_skewness_frequencies[message.id] = 1
            else:
                no_skewness_frequencies[message.id] += 1
        no_skewness_values = list(no_skewness_frequencies.values())

        no_skewness_n = len(no_skewness_values)
        no_skewness_mean = math.fsum(no_skewness_values) / no_skewness_n
        no_skewness_variance = 1 / no_skewness_n * (no_skewness_values[0] - no_skewness_mean) ** 2 + \
                   1 / no_skewness_n * (no_skewness_values[1] - no_skewness_mean) ** 2 + \
                   1 / no_skewness_n * (no_skewness_values[2] - no_skewness_mean) ** 2
        if no_skewness_variance != 0:
            no_skewness_values.sort()
            no_skewness_median = no_skewness_values[math.floor(len(no_skewness_values) / 2)]
            actual_no_skewness_result = (3 * (no_skewness_mean - no_skewness_median)) / math.sqrt(variance)
        else:
            actual_no_skewness_result = 0

        # Action
        expected_skewness_result = id_based_datasets.calculate_skewness_id_frequency(messages)
        expected_no_skewness_result = id_based_datasets.calculate_skewness_id_frequency(no_skewness_messages)

        # Assert
        self.assertEqual(expected_skewness_result, actual_skewness_result)
        self.assertEqual(expected_no_skewness_result, actual_no_skewness_result)

    def test_calculate_variance_id_frequency(self):

        # Assume
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x00\x00\x01\x01\x00\x00\x00'))
        message_3 = Message(4.000000, 130, 0, 8, bytearray(b'\x00\x00\x11\x01\x01\x01\x01\x01'))
        message_4 = Message(8.000000, 130, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        messages = [message_1, message_2, message_3, message_4]
        frequencies = {}

        for message in messages:
            if message.id not in frequencies:
                frequencies[message.id] = 1
            else:
                frequencies[message.id] += 1
        values = list(frequencies.values())

        n = len(values)
        mean = math.fsum(values) / n
        actual_variance_result = 1 / n * (values[0] - mean) ** 2 + 1 / n * (values[1] - mean) ** 2 + 1 / n * (
                values[2] - mean) ** 2

        # Action
        expected_variance_result = id_based_datasets.calculate_variance_id_frequency(messages)

        # Assert
        self.assertEqual(expected_variance_result, actual_variance_result)

    def test_calculate_kurtosis_id_frequency(self):
        # Assume
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x00\x00\x01\x01\x00\x00\x00'))
        message_3 = Message(4.000000, 130, 0, 8, bytearray(b'\x00\x00\x11\x01\x01\x01\x01\x01'))
        message_4 = Message(8.000000, 130, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        messages = [message_1, message_2, message_3, message_4]
        frequencies = {}

        for message in messages:
            if message.id not in frequencies:
                frequencies[message.id] = 1
            else:
                frequencies[message.id] += 1
        values = list(frequencies.values())

        n = len(values)
        mean = math.fsum(values) / n
        variance = 1 / n * (values[0] - mean) ** 2 + 1 / n * (values[1] - mean) ** 2 + \
                   1 / n * (values[2] - mean) ** 2

        deviation = 0
        for elem in values:
            deviation += (elem - mean) ** 4

        if variance != 0:
            actual_kurtosis_frequency_result = 1 / len(values) * deviation / variance ** 2
        else:
            actual_kurtosis_frequency_result = 0

        # Kurtosis cannot be calculated for same frequencies for all messages, due to having no variance.
        # Should result to 0.
        message_5 = Message(0.000000, 128, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        message_6 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x00\x00\x01\x01\x00\x00\x00'))
        message_7 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x00\x11\x01\x01\x01\x01\x01'))
        no_kurtosis_messages = [message_5, message_6, message_7]
        no_kurtosis_frequencies = {}

        for message in no_kurtosis_messages:
            if message.id not in no_kurtosis_frequencies:
                no_kurtosis_frequencies[message.id] = 1
            else:
                no_kurtosis_frequencies[message.id] += 1
        no_kurtosis_values = list(no_kurtosis_frequencies.values())

        no_kurtosis_n = len(no_kurtosis_values)
        no_kurtosis_mean = math.fsum(no_kurtosis_values) / no_kurtosis_n
        no_kurtosis_variance = 1 / no_kurtosis_n * (no_kurtosis_values[0] - no_kurtosis_mean) ** 2 + \
                   1 / no_kurtosis_n * (no_kurtosis_values[1] - no_kurtosis_mean) ** 2 + \
                   1 / no_kurtosis_n * (no_kurtosis_values[2] - no_kurtosis_mean) ** 2
        no_kurtosis_deviation = 0
        for elem in no_kurtosis_values:
            no_kurtosis_deviation += (elem - no_kurtosis_mean) ** 4

        if no_kurtosis_variance != 0:
            actual_no_kurtosis_frequency_result = 1 / len(no_kurtosis_values) * no_kurtosis_deviation / no_kurtosis_variance ** 2
        else:
            actual_no_kurtosis_frequency_result = 3

        # Action
        expected_kurtosis_frequency_result = id_based_datasets.calculate_kurtosis_id_frequency(messages)
        expected_no_kurtosis_frequency_result = id_based_datasets.calculate_kurtosis_id_frequency(no_kurtosis_messages)

        # Assert
        self.assertEqual(expected_kurtosis_frequency_result, actual_kurtosis_frequency_result)
        self.assertEqual(expected_no_kurtosis_frequency_result, actual_no_kurtosis_frequency_result)

    def test_calculate_kurtosis_variance_data_bit_count_id(self):

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

        n = len(variance_values)
        mean = math.fsum(variance_values) / n
        variance = 1 / n * (variance_values[0] - mean) ** 2 + 1 / n * (variance_values[1] - mean) ** 2 + \
                   1 / n * (variance_values[2] - mean) ** 2

        deviation = 0
        for elem in variance_values:
            deviation += (elem - mean) ** 4

        if variance != 0:
            actual_kurtosis_variance_bit_result = 1 / len(variance_values) * deviation / variance ** 2
        else:
            actual_kurtosis_variance_bit_result = 3

        # Action
        expected_kurtosis_result = id_based_datasets.calculate_kurtosis_variance_data_bit_count_id(messages)

        # Assert
        self.assertAlmostEqual(expected_kurtosis_result, actual_kurtosis_variance_bit_result)

    def test_calculate_num_id_transitions(self):

        # Assume
        message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\x01\x01\x01\x01\x01\x01\x01\x01'))
        message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x00\x00\x01\x01\x00\x00\x00'))
        messages = [message_1, message_2]
        messages_no_transition = []

        transitions = set()
        previous_id = messages[0].id

        if not messages:
            return 0
        else:
            for message in messages[1:]:
                transitions.add((previous_id, message.id))
                previous_id = message.id
            actual_transition_id_result = len(transitions)

        if not messages_no_transition:
            return 0
        else:
            for message in messages_no_transition[1:]:
                transitions.add((previous_id, message.id))
                previous_id = message.id
            actual_no_transition_id_result = len(transitions)

        # Action
        expected_transition_id_result = id_based_datasets.calculate_num_id_transitions(messages)
        expected_no_transition_id_result = id_based_datasets.calculate_num_id_transitions(messages_no_transition)

        # Assert
        self.assertEqual(expected_transition_id_result, actual_transition_id_result)
        self.assertEqual(expected_no_transition_id_result, actual_no_transition_id_result)


"""
    def test_calculate_req_to_res_time_variance(self):
        # Action
        result = id_based_datasets.calculate_req_to_res_time_variance(self.messages)

        # Assert
        self.assertEqual(result, 0)

    def test_calculate_kurtosis_req_to_res_time(self):
        # Action
        result = id_based_datasets.calculate_kurtosis_req_to_res_time(self.messages)

        # Assert
        self.assertAlmostEqual(result, 3, places=0)

    def calculate_req_to_res_time_variance(messages):
        intervals = []
        latest_remote_frame_timestamp = {}

        for message in messages:
            if message.add == 0b100:
                latest_remote_frame_timestamp[message.id] = message.timestamp
            elif message.add == 0b000 and latest_remote_frame_timestamp.get(message.id, None) is not None:
                intervals.append(message.timestamp - latest_remote_frame_timestamp[message.id])
                latest_remote_frame_timestamp[message.id] = None

        return 0 if len(intervals) == 0 else __calculate_variance(intervals)

    def calculate_mean_probability_bits(messages):
        bits = __calculate_probability_bits(messages)

        return math.fsum(bits) / 64.0

    def neutralize_offset(messages):
        offset = messages[0].timestamp

        for message in messages:
            message.timestamp -= offset

        return messages

    def concat_idpoints(idpoints1, idpoints2):
        offset = idpoints1[len(idpoints1) - 1].time_ms

        for idpoint in idpoints2:
            idpoint.time_ms += offset

        return idpoints1 + idpoints2

    def offset_idpoint(idp, offset):
        idp.time_ms += offset

        return idp

    def save_datasets():
        training_set, test_set = get_mixed_datasets(100, True)
        write_idpoints_csv(training_set, 100, "mixed_training")
        write_idpoints_csv(test_set, 100, "mixed_test")

    def get_mixed_datasets(period_ms=100, shuffle=True, overlap_ms=100):
        attack_free_messages1 = neutralize_offset(datareader_csv.load_attack_free1())
        attack_free_messages2 = neutralize_offset(datareader_csv.load_attack_free2())
        dos_messages = neutralize_offset(datareader_csv.load_dos())
        fuzzy_messages = neutralize_offset(datareader_csv.load_fuzzy())
        imp_messages1 = neutralize_offset(datareader_csv.load_impersonation_1())
        imp_messages2 = neutralize_offset(datareader_csv.load_impersonation_2())
        imp_messages3 = neutralize_offset(datareader_csv.load_impersonation_3())

        raw_msgs = [
            (attack_free_messages1, "normal", "attack_free_1"),
            (attack_free_messages2, "normal", "attack_free_2"),
            (dos_messages, "dos", "dos"),
            (fuzzy_messages, "fuzzy", "fuzzy"),
            (imp_messages1[0:517000], "normal", "impersonation_normal_1"),
            (imp_messages1[517000:], "impersonation", "impersonation_attack_1"),
            (imp_messages2[0:330000], "normal", "impersonation_normal_2"),
            (imp_messages2[330000:], "impersonation", "impersonation_attack_2"),
            (imp_messages3[0:534000], "normal", "impersonation_normal_3"),
            (imp_messages3[534000:], "impersonation", "impersonation_attack_3")
        ]

        datasets = []

        with conf.ProcessPoolExecutor() as executor:
            futures = {executor.submit(messages_to_idpoints, tup[0], period_ms, tup[1], overlap_ms, tup[2]) for tup in
                       raw_msgs}

            for future in conf.as_completed(futures):
                datasets.append(future.result())

        offset = 0
        points = []

        for set in datasets:
            time_low = set[0].time_ms
            points += [offset_idpoint(idp, offset - time_low) for idp in set]
            offset = points[len(points) - 1].time_ms

        training, test = train_test_split(points, shuffle=shuffle, train_size=0.8, test_size=0.2, random_state=2019)

        return training, test

    def messages_to_idpoints(messages, period_ms, is_injected, overlap_ms, name=""):
        if len(messages) == 0:
            return []

        idpoints = []
        working_set = deque()

        working_set.append(messages[0])
        lowest_index = 0
        length = len(messages)

        while lowest_index < length and (
                messages[lowest_index].timestamp * 1000.0 - working_set[0].timestamp * 1000.0) <= period_ms:
            lowest_index += 1
            working_set.append(messages[lowest_index])

        old_progress = -5
        lowest_index += 1
        highest_index = lowest_index

        for i in range(lowest_index, length):
            working_set.append(messages[i])
            progress = math.ceil((i / length) * 100.0)
            time_expended = (working_set[highest_index].timestamp - working_set[0].timestamp) * 1000.0
            highest_index += 1

            if progress % 5 == 0 and progress > old_progress:
                print(f"{name} Creating idpoints: {progress}/100%")
                old_progress = progress

            if time_expended >= overlap_ms:
                low = working_set.popleft()

                while (messages[i].timestamp * 1000.0 - low.timestamp * 1000.0) > period_ms:
                    low = working_set.popleft()
                    highest_index -= 1

                working_set.appendleft(low)
                idpoints.append(messages_to_idpoint(list(working_set), is_injected))

        return idpoints

    def messages_to_idpoint(messages, is_injected):
        # this function may never be called with an empty list

        # maps a function to an attribute. The function must accept a list of messages.
        # missing mappings are allowed, and will give the feature a value of 0
        attribute_function_mappings = {
            "time_ms": lambda msgs: msgs[0].timestamp * 1000,
            "is_injected": lambda msgs: is_injected,
            "mean_id_interval": calculate_mean_id_interval,
            "variance_id_frequency": calculate_variance_id_frequency,
            "num_id_transitions": calculate_num_id_transitions,
            "num_ids": calculate_num_ids,
            "num_msgs": len,
            "mean_id_intervals_variance": calculate_mean_id_intervals_variance,
            "mean_data_bit_count": calculate_mean_data_bit_count,
            "variance_data_bit_count": calculate_variance_data_bit_count,
            "mean_variance_data_bit_count_id": calculate_mean_variance_data_bit_count_id,
            "mean_probability_bits": calculate_mean_probability_bits,
            "req_to_res_time_variance": calculate_req_to_res_time_variance,
            "kurtosis_id_interval": calculate_kurtosis_id_interval,
            "kurtosis_id_frequency": calculate_kurtosis_id_frequency,
            "kurtosis_mean_id_intervals": calculate_kurtosis_mean_id_intervals,
            "kurtosis_variance_data_bit_count_id": calculate_kurtosis_variance_data_bit_count_id,
            "skewness_id_interval_variances": calculate_skewness_id_interval_variances,
            "skewness_id_frequency": calculate_skewness_id_frequency,
            "kurtosis_req_to_res_time": calculate_kurtosis_req_to_res_time
        }

        # Blank idpoint
        idpoint = idp.IDPoint(*[0 for attr in idp.idpoint_attributes])

        # Update blank idpoint from attribute functions
        for attr in idp.idpoint_attributes:
            feature_func = attribute_function_mappings.get(attr, None)

            if feature_func is not None:
                setattr(idpoint, attr, feature_func(messages))

        return idpoint
"""