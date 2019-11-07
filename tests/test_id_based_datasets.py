import math
from unittest import TestCase
import datasets
from datapoint import IDPoint
from message import Message


class TestIdBasedDatasets(TestCase):
    def setUp(self) -> None:
        # Assume for calculate_mean_id_intervals_variance, calculate_skewness_id_interval_variances
        self.message_1 = Message(0.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        self.message_2 = Message(2.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        self.message_3 = Message(4.000000, 128, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00\x00\x53'))
        self.message_4 = Message(6.000000, 129, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        self.message_5 = Message(8.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        self.message_6 = Message(10.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        message_7 = Message(12.000000, 129, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        message_8 = Message(14.000000, 128, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        message_9 = Message(16.000000, 130, 0, 8, bytearray(b'\x00\x29\x20\x00\x00\x45\x00\x00'))
        self.messages = [self.message_1, self.message_2, self.message_3, self.message_4, self.message_5, self.message_6,
                         message_7, message_8, message_9]

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

        # Assum for message_to_idpoint
        time_ms = 0.0
        class_label = "normal"
        mean_id_interval = datasets.calculate_mean_id_interval(self.messages)
        variance_id_frequency = datasets.calculate_variance_id_frequency(self.messages)
        num_id_transitions = datasets.calculate_num_id_transitions(self.messages)
        num_ids = datasets.calculate_num_ids(self.messages)
        num_msgs = len(self.messages)
        mean_id_interval_variance = datasets.calculate_mean_id_intervals_variance(self.messages)
        mean_data_bit_count = datasets.calculate_mean_data_bit_count(self.messages)
        variance_data_bit_count = datasets.calculate_variance_data_bit_count(self.messages)
        mean_variance_data_bit_count = datasets.calculate_mean_variance_data_bit_count_id(self.messages)
        mean_probability_bits = datasets.calculate_mean_probability_bits(self.messages)
        req_to_res_time_variance = datasets.calculate_req_to_res_time_variance(self.messages)
        kurtosis_id_interval = datasets.calculate_kurtosis_id_interval(self.messages)
        kurtosis_id_frequency = datasets.calculate_kurtosis_id_frequency(self.messages)
        kurtosis_mean_id_intervals = datasets.calculate_kurtosis_mean_id_intervals(self.messages)
        kurtosis_variance_data_bit_count_id = datasets.calculate_kurtosis_variance_data_bit_count_id(
            self.messages)
        skewness_id_interval_variances = datasets.calculate_skewness_id_interval_variances(self.messages)
        skewness_id_frequency = datasets.calculate_skewness_id_frequency(self.messages)
        kurtosis_req_to_res_time = datasets.calculate_kurtosis_req_to_res_time(self.messages)

        self.actual_datapoint = IDPoint(time_ms,
                                   class_label,
                                   mean_id_interval,
                                   variance_id_frequency,
                                   num_id_transitions,
                                   num_ids,
                                   num_msgs,
                                   mean_id_interval_variance,
                                   mean_data_bit_count,
                                   variance_data_bit_count,
                                   mean_variance_data_bit_count,
                                   mean_probability_bits,
                                   req_to_res_time_variance,
                                   kurtosis_id_interval,
                                   kurtosis_id_frequency,
                                   skewness_id_frequency,
                                   kurtosis_mean_id_intervals,
                                   kurtosis_variance_data_bit_count_id,
                                   skewness_id_interval_variances,
                                   kurtosis_req_to_res_time)

        # Assume for test_offset_idpoint,
        self.time_ms = 1881258.816042923
        self.class_label = "normal"
        self.mean_id_interval = 0.011034680628272332,
        self.variance_id_frequency = 17.72437499999999
        self.num_id_transitions = 101
        self.num_ids = 40
        self.num_msgs = 231
        self.mean_id_intervals_variance = 8.354867823557838e-05
        self.mean_data_bit_count = 14.913419913419913
        self.variance_data_bit_count = 74.46869436479822
        self.mean_variance_data_bit_count_id = 4.239600442036318
        self.mean_probability_bits = 0.23302218614718614
        self.req_to_res_time_variance = 9.750100002858036e-10
        self.kurtosis_id_interval = 74.39655141184994
        self.kurtosis_id_frequency = 1.323601387295182
        self.skewness_id_frequency = 0.5522522478213545
        self.kurtosis_mean_id_intervals = 5.511232560404617
        self.kurtosis_variance_data_bit_count_id = 15.457159141707956
        self.skewness_id_interval_variances = 0.694951701467612
        self.kurtosis_req_to_res_time = 6.3109383892063535

        self.id_point_object = IDPoint(self.time_ms, self.class_label,
                                       self.mean_id_interval,
                                       self.variance_id_frequency,
                                       self.num_id_transitions,
                                       self.num_ids, self.num_msgs,
                                       self.mean_id_intervals_variance,
                                       self.mean_data_bit_count,
                                       self.variance_data_bit_count,
                                       self.mean_variance_data_bit_count_id,
                                       self.mean_probability_bits,
                                       self.req_to_res_time_variance,
                                       self.kurtosis_id_interval,
                                       self.kurtosis_id_frequency,
                                       self.skewness_id_frequency,
                                       self.kurtosis_mean_id_intervals,
                                       self.kurtosis_variance_data_bit_count_id,
                                       self.skewness_id_interval_variances,
                                       self.kurtosis_req_to_res_time)

    def test_calculate_num_ids(self):

        # Assume
        actual_ids_seen = set()

        for message in self.messages:
            actual_ids_seen.add(message.id)

        # Action
        expected_result = datasets.calculate_num_ids(self.messages)

        # Assert
        self.assertEqual(expected_result, len(actual_ids_seen))

    def test_neutralize_offset(self):
        # Assume
        message_1 = Message(4.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_2 = Message(8.000000, 128, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_3 = Message(12.000000, 130, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00\x00\x53'))
        messages = [message_1, message_2, message_3]

        # Offset is calculated as
        offset = messages[0].timestamp  # which is the first and only element, with timestamp 12.0

        # Messages in messages_2 will be offset with the timestamp in message_1. Thus, 4 + 12, 6 + 12, 8 + 12.
        actual_offset_message_1 = messages[0].timestamp - offset
        actual_offset_message_2 = messages[1].timestamp - offset
        actual_offset_message_3 = messages[2].timestamp - offset

        # Action
        expected_offset_result = datasets.__neutralize_offset(messages)

        # Assert
        self.assertEqual(expected_offset_result, messages)
        self.assertEqual(expected_offset_result[0].timestamp, actual_offset_message_1)
        self.assertEqual(expected_offset_result[1].timestamp, actual_offset_message_2)
        self.assertEqual(expected_offset_result[2].timestamp, actual_offset_message_3)

    def test_offset_idpoint(self):

        # Arbitrary offset
        offset = 4
        self.id_point_object.time_ms += offset

        actual_id_point_offset = self.id_point_object

        # Action
        expected_result = datasets.offset_idpoint(self.id_point_object, offset)

        # Assert
        self.assertEqual(expected_result, actual_id_point_offset)

    def test_concat_messages(self):

        # Assume
        message_1 = Message(12.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        messages_1 = [message_1]

        message_2 = Message(4.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_3 = Message(6.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_4 = Message(8.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        messages_2 = [message_2, message_3, message_4]

        message_2_duplicate = Message(4.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_3_duplicate = Message(6.000000, 129, 0, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_4_duplicate = Message(8.000000, 128, 0, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))

        # Duplicate used to concatenate
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
        expected_result = datasets.__concat_messages(messages_1, messages_2)

        # Assert
        self.assertEqual(expected_result, actual_concat_messages_result)
        self.assertEqual(expected_result[1].timestamp, actual_offset_message_2)
        self.assertEqual(expected_result[2].timestamp, actual_offset_message_3)
        self.assertEqual(expected_result[3].timestamp, actual_offset_message_4)

    def test_concat_idpoints(self):
        #Two separate timestamps used when concatenating by offestting them from the first list of idpoints
        time_ms_2 = 2011258.112002223
        time_ms_3 = 2704818.516082299

        id_points = [self.id_point_object]

        id_point_object_2 = IDPoint(time_ms_2,
                                    self.class_label,
                                    self.mean_id_interval,
                                    self.variance_id_frequency,
                                    self.num_id_transitions,
                                    self.num_ids,
                                    self.num_msgs,
                                    self.mean_id_intervals_variance,
                                    self.mean_data_bit_count,
                                    self.variance_data_bit_count,
                                    self.mean_variance_data_bit_count_id,
                                    self.mean_probability_bits,
                                    self.req_to_res_time_variance,
                                    self.kurtosis_id_interval,
                                    self.kurtosis_id_frequency,
                                    self.skewness_id_frequency,
                                    self.kurtosis_mean_id_intervals,
                                    self.kurtosis_variance_data_bit_count_id,
                                    self.skewness_id_interval_variances,
                                    self.kurtosis_req_to_res_time)

        id_point_object_3 = IDPoint(time_ms_3,
                                    self.class_label,
                                    self.mean_id_interval,
                                    self.variance_id_frequency,
                                    self.num_id_transitions,
                                    self.num_ids, self.num_msgs,
                                    self.mean_id_intervals_variance,
                                    self.mean_data_bit_count,
                                    self.variance_data_bit_count,
                                    self.mean_variance_data_bit_count_id,
                                    self.mean_probability_bits,
                                    self.req_to_res_time_variance,
                                    self.kurtosis_id_interval,
                                    self.kurtosis_id_frequency,
                                    self.skewness_id_frequency,
                                    self.kurtosis_mean_id_intervals,
                                    self.kurtosis_variance_data_bit_count_id,
                                    self.skewness_id_interval_variances,
                                    self.kurtosis_req_to_res_time)

        id_points_2 = [id_point_object_2, id_point_object_3]

        id_point_object_2_dup = IDPoint(time_ms_2,
                                    self.class_label,
                                    self.mean_id_interval,
                                    self.variance_id_frequency,
                                    self.num_id_transitions,
                                    self.num_ids,
                                    self.num_msgs,
                                    self.mean_id_intervals_variance,
                                    self.mean_data_bit_count,
                                    self.variance_data_bit_count,
                                    self.mean_variance_data_bit_count_id,
                                    self.mean_probability_bits,
                                    self.req_to_res_time_variance,
                                    self.kurtosis_id_interval,
                                    self.kurtosis_id_frequency,
                                    self.skewness_id_frequency,
                                    self.kurtosis_mean_id_intervals,
                                    self.kurtosis_variance_data_bit_count_id,
                                    self.skewness_id_interval_variances,
                                    self.kurtosis_req_to_res_time)

        id_point_object_3_dup = IDPoint(time_ms_3,
                                    self.class_label,
                                    self.mean_id_interval,
                                    self.variance_id_frequency,
                                    self.num_id_transitions,
                                    self.num_ids, self.num_msgs,
                                    self.mean_id_intervals_variance,
                                    self.mean_data_bit_count,
                                    self.variance_data_bit_count,
                                    self.mean_variance_data_bit_count_id,
                                    self.mean_probability_bits,
                                    self.req_to_res_time_variance,
                                    self.kurtosis_id_interval,
                                    self.kurtosis_id_frequency,
                                    self.skewness_id_frequency,
                                    self.kurtosis_mean_id_intervals,
                                    self.kurtosis_variance_data_bit_count_id,
                                    self.skewness_id_interval_variances,
                                    self.kurtosis_req_to_res_time)

        id_points_2_dup = [id_point_object_2_dup, id_point_object_3_dup]

        # Offset is calculated as
        offset = id_points[len(id_points) - 1].time_ms

        # Messages in messages_2 will be offset with the timestamp in message_1. Thus, 4 + 12, 6 + 12, 8 + 12.
        actual_offset_idpoint_2 = id_points_2[0].time_ms + offset
        actual_offset_idpoint_3 = id_points_2[1].time_ms + offset

        # Concatenate the messages in messages_1 onto messages_2_duplicate (which is the same as messages_2)
        for idp in id_points_2_dup:
            idp.time_ms += offset
        actual_concat_point_result = id_points + id_points_2_dup

        # Action
        expected_result = datasets.concat_idpoints(id_points, id_points_2)

        # Assert
        self.assertEqual(expected_result, actual_concat_point_result)
        self.assertEqual(expected_result[1].time_ms, actual_offset_idpoint_2)
        self.assertEqual(expected_result[2].time_ms, actual_offset_idpoint_3)

    def test_calculate_skewness_id_interval_variances(self):

        # Assumption when skewness can be calculated for a set of messages
        n_final = len(self.values)
        mean_final = (self.values[0] + self.values[1] + self.values[2]) / n_final
        variance_final = (1 / n_final) * (self.values[0] - mean_final) ** 2 + (1 / n_final) * \
                         (self.values[1] - mean_final) ** 2 + (1 / n_final) * (self.values[2] - mean_final) ** 2
        actual_skewness_result = (3 * (mean_final - self.median)) / math.sqrt(variance_final)

        # Assumption when skewness can NOT be calculated when unique ids is required to appear at least 3 times
        no_skewness_messages = [self.message_1, self.message_2, self.message_3, self.message_4, self.message_5,
                                self.message_6]

        # Action
        expected_result = datasets.calculate_skewness_id_interval_variances(self.messages)
        expected_no_skewness_result = datasets.calculate_skewness_id_interval_variances(no_skewness_messages)

        # Assert
        self.assertEqual(expected_result, actual_skewness_result)
        self.assertEqual(expected_no_skewness_result, 0)

    def test_calculate_mean_id_intervals_variance(self):

        # Assume
        actual_result = math.fsum(self.values) / len(self.values)

        # If length of interval variances is 0, we can't calculate anything
        # Assumption when skewness can NOT be calculated when unique ids is required to appear at least 3 times
        no_mean_messages = [self.message_1, self.message_2, self.message_3, self.message_4, self.message_5,
                            self.message_6]

        # Action
        expected_result = datasets.calculate_mean_id_intervals_variance(self.messages)
        expected_no_mean_result = datasets.calculate_skewness_id_interval_variances(no_mean_messages)

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
        expected_result = datasets.calculate_mean_id_interval(messages)
        expected_no_mean_id_interval_messages = datasets.calculate_mean_id_interval(no_mean_id_interval_messages)

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
        expected_kurtosis_result = datasets.calculate_kurtosis_id_interval(self.messages)

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
        expected_kurtosis_mean_result = datasets.calculate_kurtosis_mean_id_intervals(self.messages)

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
        expected_result = datasets.calculate_variance_data_bit_count(messages)
        expected_no_bit_count = datasets.calculate_variance_data_bit_count(no_bit_count_messages)
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

        # Assume we can calculate the mean variance data bit count
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
        expected_result = datasets.calculate_mean_variance_data_bit_count_id(messages)
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
        expected_mean_result = datasets.calculate_mean_data_bit_count(messages)

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
        expected_skewness_result = datasets.calculate_skewness_id_frequency(messages)
        expected_no_skewness_result = datasets.calculate_skewness_id_frequency(no_skewness_messages)

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
        expected_variance_result = datasets.calculate_variance_id_frequency(messages)

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
        expected_kurtosis_frequency_result = datasets.calculate_kurtosis_id_frequency(messages)
        expected_no_kurtosis_frequency_result = datasets.calculate_kurtosis_id_frequency(no_kurtosis_messages)

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
        expected_kurtosis_result = datasets.calculate_kurtosis_variance_data_bit_count_id(messages)

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
        expected_transition_id_result = datasets.calculate_num_id_transitions(messages)
        expected_no_transition_id_result = datasets.calculate_num_id_transitions(messages_no_transition)

        # Assert
        self.assertEqual(expected_transition_id_result, actual_transition_id_result)
        self.assertEqual(expected_no_transition_id_result, actual_no_transition_id_result)

    def test_calculate_req_to_res_time_variance(self):

        # Assume
        message_1 = Message(0.000000, 128, 4, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_2 = Message(2.000000, 129, 4, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_3 = Message(6.000000, 128, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00\x00\x53'))
        message_4 = Message(10.000000, 129, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        message_5 = Message(11.000000, 130, 4, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        message_6 = Message(20.000000, 130, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        messages = [message_1, message_2, message_3, message_4, message_5, message_6]

        intervals = []
        recent_remote_frame_timestamp = {}

        for message in messages:
            if message.rtr == 0b100:
                recent_remote_frame_timestamp[message.id] = message.timestamp
            elif message.rtr == 0b000 and recent_remote_frame_timestamp.get(message.id, None) is not None:
                intervals.append(message.timestamp - recent_remote_frame_timestamp[message.id])
                recent_remote_frame_timestamp[message.id] = None

        if len(intervals) == 0:
            actual_req_to_res_variance_result = 0
        else:
            n = len(intervals)
            mean = (intervals[0] + intervals[1] + intervals[2]) / n
            variance = 1 / n * (intervals[0] - mean) ** 2 + 1 / n * (intervals[1] - mean) ** 2 + 1 / n * (
                    intervals[2] - mean) ** 2
            actual_req_to_res_variance_result = variance


        # Action
        expected_req_to_res_variance_result = datasets.calculate_req_to_res_time_variance(messages)

        # Assert
        self.assertEqual(expected_req_to_res_variance_result, actual_req_to_res_variance_result)

    def test_calculate_kurtosis_req_to_res_time(self):

        # Assume
        message_1 = Message(0.000000, 128, 4, 8, bytearray(b'\xd7\xa7\x7f\x8c\x11\x2f\x00\x10'))
        message_2 = Message(2.000000, 129, 4, 8, bytearray(b'\x00\x17\xea\x0a\x20\x1a\x20\x43'))
        message_3 = Message(6.000000, 128, 0, 8, bytearray(b'\x7f\x84\x60\x00\x00\x00\x00\x53'))
        message_4 = Message(10.000000, 129, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        message_5 = Message(10.000000, 130, 4, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        message_6 = Message(20.000000, 130, 0, 8, bytearray(b'\x00\x80\x10\xff\x00\xff\x40\xce'))
        messages = [message_1, message_2, message_3, message_4, message_5, message_6]

        intervals = []
        recent_remote_frame_timestamp = {}

        for message in messages:
            if message.rtr == 0b100:
                recent_remote_frame_timestamp[message.id] = message.timestamp
            elif message.rtr == 0b000 and recent_remote_frame_timestamp.get(message.id, None) is not None:
                intervals.append(message.timestamp - recent_remote_frame_timestamp[message.id])
                recent_remote_frame_timestamp[message.id] = None
        n = len(intervals)
        mean = (intervals[0] + intervals[1] + intervals[2]) / n
        variance = 1 / n * (intervals[0] - mean) ** 2 + 1 / n * (intervals[1] - mean) ** 2 + 1 / n * (
                    intervals[2] - mean) ** 2

        deviation = 0
        for elem in intervals:
            deviation += (elem - mean) ** 4

        if variance != 0:
            actual_req_to_res_kurtosis_result = 1 / len(intervals) * deviation / variance ** 2
        else:
            actual_req_to_res_kurtosis_result = 3

        # Action
        expected_req_to_res_kurtosis_result = datasets.calculate_kurtosis_req_to_res_time(messages)

        # Assert
        self.assertEqual(expected_req_to_res_kurtosis_result, actual_req_to_res_kurtosis_result)

    def test_messages_to_idpoint(self):

        # Action
        expected_value = datasets.messages_to_idpoint(self.messages, "normal")
        #expected_exception_raise = id_based_datasets.messages_to_idpoint([], "normal")

        # Assert
        self.assertAlmostEqual(expected_value.time_ms, self.actual_datapoint.time_ms)
        self.assertAlmostEqual(expected_value.class_label, self.actual_datapoint.class_label)
        self.assertAlmostEqual(expected_value.mean_id_interval, self.actual_datapoint.mean_id_interval)
        self.assertAlmostEqual(expected_value.variance_id_frequency, self.actual_datapoint.variance_id_frequency)
        self.assertAlmostEqual(expected_value.num_id_transitions, self.actual_datapoint.num_id_transitions)
        self.assertAlmostEqual(expected_value.num_ids, self.actual_datapoint.num_ids)
        self.assertAlmostEqual(expected_value.num_msgs, self.actual_datapoint.num_msgs)
        self.assertAlmostEqual(expected_value.mean_id_intervals_variance, self.actual_datapoint.mean_id_intervals_variance)
        self.assertAlmostEqual(expected_value.mean_data_bit_count, self.actual_datapoint.mean_data_bit_count)
        self.assertAlmostEqual(expected_value.variance_data_bit_count, self.actual_datapoint.variance_data_bit_count)
        self.assertAlmostEqual(expected_value.mean_variance_data_bit_count_id, self.actual_datapoint.mean_variance_data_bit_count_id)
        self.assertAlmostEqual(expected_value.mean_probability_bits, self.actual_datapoint.mean_probability_bits)
        self.assertAlmostEqual(expected_value.req_to_res_time_variance, self.actual_datapoint.req_to_res_time_variance)
        self.assertAlmostEqual(expected_value.kurtosis_id_interval, self.actual_datapoint.kurtosis_id_interval)
        self.assertAlmostEqual(expected_value.kurtosis_id_frequency, self.actual_datapoint.kurtosis_id_frequency)
        self.assertAlmostEqual(expected_value.kurtosis_mean_id_intervals, self.actual_datapoint.kurtosis_mean_id_intervals)
        self.assertAlmostEqual(expected_value.kurtosis_variance_data_bit_count_id, self.actual_datapoint.kurtosis_variance_data_bit_count_id)
        self.assertAlmostEqual(expected_value.skewness_id_interval_variances, self.actual_datapoint.skewness_id_interval_variances)
        self.assertAlmostEqual(expected_value.skewness_id_frequency, self.actual_datapoint.skewness_id_frequency)
        self.assertAlmostEqual(expected_value.kurtosis_req_to_res_time, self.actual_datapoint.kurtosis_req_to_res_time)
        #self.assertRaises(ValueError, expected_exception_raise)