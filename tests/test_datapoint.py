from unittest import TestCase
import datapoint
from datapoint import DataPoint


class TestDataPoint(TestCase):
    def setUp(self) -> None:
        time_ms = 1881258.816042923
        class_label = "normal"
        mean_id_interval = 0.011034680628272332,
        variance_id_frequency = 17.72437499999999
        num_id_transitions = 101
        num_ids = 40
        num_msgs = 231
        mean_id_intervals_variance = 8.354867823557838e-05
        mean_data_bit_count = 14.913419913419913
        variance_data_bit_count = 74.46869436479822
        mean_variance_data_bit_count_id = 4.239600442036318
        kurtosis_id_interval = 74.39655141184994
        kurtosis_id_frequency = 1.323601387295182
        skewness_id_frequency = 0.5522522478213545
        kurtosis_mean_id_intervals = 5.511232560404617
        kurtosis_variance_data_bit_count_id = 15.457159141707956
        skewness_id_interval_variances = 0.694951701467612

        self.data_point_row = [time_ms, class_label, mean_id_interval, variance_id_frequency, num_id_transitions, num_ids,
                               num_msgs, mean_id_intervals_variance, mean_data_bit_count, variance_data_bit_count,
                               mean_variance_data_bit_count_id, kurtosis_id_interval, kurtosis_id_frequency,
                               skewness_id_frequency, kurtosis_mean_id_intervals, kurtosis_variance_data_bit_count_id,
                               skewness_id_interval_variances]

        self.data_point_object = DataPoint(time_ms, class_label, mean_id_interval, variance_id_frequency,
                                           num_id_transitions, num_ids, num_msgs, mean_id_intervals_variance,
                                           mean_data_bit_count, variance_data_bit_count,
                                           mean_variance_data_bit_count_id, kurtosis_id_interval,
                                           kurtosis_id_frequency, skewness_id_frequency, kurtosis_mean_id_intervals,
                                           kurtosis_variance_data_bit_count_id, skewness_id_interval_variances)

    def test_parse_csv_row(self):
        result = datapoint.parse_csv_row(self.data_point_row)

        self.assertEqual(result, self.data_point_object)

    def test_get_csv_row(self):
        result = datapoint.get_csv_row(self.data_point_object)

        self.assertEqual(result, self.data_point_row)
