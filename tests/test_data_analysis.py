from unittest import TestCase
import data_analysis
import pandas as pd
import os
import datareader_csv


class TestDataAnalysis(TestCase):
    def setUp(self) -> None:
        os.chdir("..")

        self.messages = datareader_csv.load_messages("data/csv/Attack_free_dataset.csv", limit=100)

    def test_get_mean_time_between_normal_messages(self):
        # Using a subset of the messages read from the Attack_free_dataset file.
        messages = self.messages[0:10]
        actual_result = data_analysis.get_mean_time_between_normal_messages(messages)

        self.assertEqual(0.0002285714285714286, actual_result)

    def test_get_mean_time_between_split_messages(self):
        actual_result = data_analysis.get_mean_time_between_split_messages(self.messages)

        self.assertEqual(0.00041562, round(actual_result, 8))
