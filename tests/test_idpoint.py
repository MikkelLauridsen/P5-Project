from unittest import TestCase
import idpoint
from idpoint import IDPoint

class TestIdpoint(TestCase):

    def test_parse_csv_row(self):
        # Assume
        time_ms = 1881258.816042923
        is_injected = "normal"
        mean_id_interval = 0.011034680628272332,
        variance_id_frequency = 17.72437499999999
        num_id_transitions = 101
        num_ids = 40
        num_msgs = 231
        mean_id_intervals_variance = 8.354867823557838e-05
        mean_data_bit_count = 14.913419913419913
        variance_data_bit_count = 74.46869436479822
        mean_variance_data_bit_count_id = 4.239600442036318
        mean_probability_bits = 0.23302218614718614
        req_to_res_time_variance = 9.750100002858036e-10
        kurtosis_id_interval = 74.39655141184994
        kurtosis_id_frequency = 1.323601387295182
        skewness_id_frequency = 0.5522522478213545
        kurtosis_mean_id_intervals = 5.511232560404617
        kurtosis_variance_data_bit_count_id = 15.457159141707956
        skewness_id_interval_variances = 0.694951701467612
        kurtosis_req_to_res_time = 6.3109383892063535
        id_point = [time_ms, is_injected, mean_id_interval, variance_id_frequency, num_id_transitions, num_ids,
                    num_msgs, mean_id_intervals_variance, mean_data_bit_count, variance_data_bit_count,
                    mean_variance_data_bit_count_id, mean_probability_bits, req_to_res_time_variance, kurtosis_id_interval,
                    kurtosis_id_frequency, skewness_id_frequency, kurtosis_mean_id_intervals, kurtosis_variance_data_bit_count_id,
                    skewness_id_interval_variances, kurtosis_req_to_res_time]

        # Action
        result = idpoint.parse_csv_row(id_point)

        # Assert
        self.assertEqual(result.time_ms, time_ms)
        self.assertEqual(result.is_injected, is_injected)
        self.assertEqual(result.mean_id_interval, mean_id_interval)
        self.assertEqual(result.variance_id_frequency, variance_id_frequency)
        self.assertEqual(result.num_id_transitions, num_id_transitions)
        self.assertTrue(result.num_ids, num_ids)
        self.assertEqual(result.num_msgs, num_msgs)
        self.assertEqual(result.mean_id_intervals_variance, mean_id_intervals_variance)
        self.assertEqual(result.mean_data_bit_count, mean_data_bit_count)
        self.assertEqual(result.variance_data_bit_count, variance_data_bit_count)
        self.assertEqual(result.mean_variance_data_bit_count_id, mean_variance_data_bit_count_id)
        self.assertEqual(result.mean_probability_bits, mean_probability_bits)
        self.assertEqual(result.req_to_res_time_variance, req_to_res_time_variance)
        self.assertEqual(result.kurtosis_id_interval, kurtosis_id_interval)
        self.assertEqual(result.kurtosis_id_frequency, kurtosis_id_frequency)
        self.assertEqual(result.skewness_id_frequency, skewness_id_frequency)
        self.assertEqual(result.kurtosis_mean_id_intervals, kurtosis_mean_id_intervals)
        self.assertEqual(result.kurtosis_variance_data_bit_count_id, kurtosis_variance_data_bit_count_id)
        self.assertEqual(result.skewness_id_interval_variances, skewness_id_interval_variances)
        self.assertEqual(result.kurtosis_req_to_res_time, kurtosis_req_to_res_time)
        self.assertTrue(result, IDPoint)

    def test_get_csv_row(self):
        # Assume
        time_ms = 1881258.816042923
        is_injected = "normal"
        mean_id_interval = 0.011034680628272332,
        variance_id_frequency = 17.72437499999999
        num_id_transitions = 101
        num_ids = 40
        num_msgs = 231
        mean_id_intervals_variance = 8.354867823557838e-05
        mean_data_bit_count = 14.913419913419913
        variance_data_bit_count = 74.46869436479822
        mean_variance_data_bit_count_id = 4.239600442036318
        mean_probability_bits = 0.23302218614718614
        req_to_res_time_variance = 9.750100002858036e-10
        kurtosis_id_interval = 74.39655141184994
        kurtosis_id_frequency = 1.323601387295182
        skewness_id_frequency = 0.5522522478213545
        kurtosis_mean_id_intervals = 5.511232560404617
        kurtosis_variance_data_bit_count_id = 15.457159141707956
        skewness_id_interval_variances = 0.694951701467612
        kurtosis_req_to_res_time = 6.3109383892063535
        id_point = IDPoint(time_ms, is_injected, mean_id_interval, variance_id_frequency, num_id_transitions, num_ids,
                    num_msgs, mean_id_intervals_variance, mean_data_bit_count, variance_data_bit_count,
                    mean_variance_data_bit_count_id, mean_probability_bits, req_to_res_time_variance,
                    kurtosis_id_interval,
                    kurtosis_id_frequency, skewness_id_frequency, kurtosis_mean_id_intervals,
                    kurtosis_variance_data_bit_count_id,
                    skewness_id_interval_variances, kurtosis_req_to_res_time)

        # Action
        result = idpoint.get_csv_row(id_point)

        # Assert
        self.assertEqual(result[0], time_ms)
        self.assertEqual(result[1], is_injected)
        self.assertEqual(result[2], mean_id_interval)
        self.assertEqual(result[3], variance_id_frequency)
        self.assertEqual(result[4], num_id_transitions)
        self.assertTrue(result[5], num_ids)
        self.assertEqual(result[6], num_msgs)
        self.assertEqual(result[7], mean_id_intervals_variance)
        self.assertEqual(result[8], mean_data_bit_count)
        self.assertEqual(result[9], variance_data_bit_count)
        self.assertEqual(result[10], mean_variance_data_bit_count_id)
        self.assertEqual(result[11], mean_probability_bits)
        self.assertEqual(result[12], req_to_res_time_variance)
        self.assertEqual(result[13], kurtosis_id_interval)
        self.assertEqual(result[14], kurtosis_id_frequency)
        self.assertEqual(result[15], skewness_id_frequency)
        self.assertEqual(result[16], kurtosis_mean_id_intervals)
        self.assertEqual(result[17], kurtosis_variance_data_bit_count_id)
        self.assertEqual(result[18], skewness_id_interval_variances)
        self.assertEqual(result[19], kurtosis_req_to_res_time)
        self.assertTrue(result, [time_ms, is_injected, mean_id_interval, variance_id_frequency, num_id_transitions, num_ids,
                    num_msgs, mean_id_intervals_variance, mean_data_bit_count, variance_data_bit_count,
                    mean_variance_data_bit_count_id, mean_probability_bits, req_to_res_time_variance,
                    kurtosis_id_interval,
                    kurtosis_id_frequency, skewness_id_frequency, kurtosis_mean_id_intervals,
                    kurtosis_variance_data_bit_count_id,
                    skewness_id_interval_variances, kurtosis_req_to_res_time])