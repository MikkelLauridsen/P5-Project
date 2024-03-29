from unittest import TestCase
import datapoint
from datapoint import DataPoint


class TestDataPoint(TestCase):
    def setUp(self) -> None:
        time_ms = 1881258.816042923
        class_label = "normal"
        mean_id_interval = 0.011034680628272332,
        variance_id_frequency = 17.72437499999999
        mean_id_intervals_variance = 8.354867823557838e-05
        mean_data_bit_count = 14.913419913419913
        variance_data_bit_count = 74.46869436479822
        kurtosis_id_interval = 74.39655141184994
        kurtosis_id_frequency = 1.323601387295182
        skewness_id_frequency = 0.5522522478213545
        kurtosis_variance_data_bit_count_id = 15.457159141707956
        skewness_id_interval_variances = 0.694951701467612

        self.data_point_row = [time_ms, class_label, mean_id_interval, variance_id_frequency,
                               mean_id_intervals_variance, mean_data_bit_count, variance_data_bit_count,
                               kurtosis_id_interval, kurtosis_id_frequency, skewness_id_frequency,
                               kurtosis_variance_data_bit_count_id, skewness_id_interval_variances]

        self.data_point_object = DataPoint(time_ms, class_label, mean_id_interval, variance_id_frequency,
                                   mean_id_intervals_variance, mean_data_bit_count, variance_data_bit_count,
                                   kurtosis_id_interval, kurtosis_id_frequency, skewness_id_frequency,
                                   kurtosis_variance_data_bit_count_id, skewness_id_interval_variances)

        self.header = datapoint.datapoint_attributes

    def test_parse_csv_row(self):
        result = datapoint.parse_csv_row(self.data_point_row)

        self.assertEqual(result, self.data_point_object)

    def test_get_csv_row(self):
        result = datapoint.get_csv_row(self.data_point_object)

        self.assertEqual(result, self.data_point_row)

    def test_is_header_matching_match(self):
        diff, diff_list = datapoint.is_header_matching(self.header)
        self.assertTrue(diff)
        self.assertEqual(diff_list, set())

    def test_is_header_matching_no_match(self):
        header = []

        diff, diff_list = datapoint.is_header_matching(header)
        self.assertFalse(diff)
        self.assertEqual(diff_list, set(self.header))
