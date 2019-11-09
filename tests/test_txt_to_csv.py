from unittest import TestCase
import os
import txt_to_csv
import shutil
import pandas as pd


class TestTxtToCsv(TestCase):
    def setUp(self) -> None:
        os.chdir("..")
        txt_dir = "tests/txt/"

        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        messages = []

        # Setting up a test txt file containing the first 10 rows in the file "Attack_free_dataset.txt".
        with open("data/raw/Attack_free_dataset.txt", "r") as reader:
            for i in range(10):
                messages.append(reader.readline())

            with open("tests/txt/Attack_free_dataset.txt", "w") as writer:
                for message in messages:
                    writer.write(message)

        messages = []

        # Setting up a test txt file containing the first 10 rows in the file "Attack_free_dataset2.txt".
        with open("data/raw/Attack_free_dataset2.txt", "r") as reader:
            for i in range(11):
                messages.append(reader.readline())
            with open("tests/txt/Attack_free_dataset2.txt", "w") as writer:
                for message in messages:
                    writer.write(message)

        self.attack_free_1_csv_expected = pd.read_csv("tests/csv_expected/Attack_free_dataset.csv").values.tolist()
        self.attack_free_2_csv_expected = pd.read_csv("tests/csv_expected/Attack_free_dataset2.csv").values.tolist()

    def tearDown(self) -> None:
        # Deleting the files used in the tests.
        delete_list = ["tests/txt/", "tests/csv_actual/"]

        for filepath in delete_list:
            if os.path.exists(filepath):
                shutil.rmtree(filepath, ignore_errors=True)

    def test_txt_to_csv(self):
        txt_to_csv.txt_to_csv("tests/txt/", "tests/csv_actual/")

        attack_free_1_csv_actual = pd.read_csv("tests/csv_actual/Attack_free_dataset.csv").values.tolist()
        attack_free_2_csv_actual = pd.read_csv("tests/csv_actual/Attack_free_dataset2.csv").values.tolist()

        self.assertEqual(self.attack_free_1_csv_expected, attack_free_1_csv_actual)
        self.assertEqual(self.attack_free_2_csv_expected, attack_free_2_csv_actual)
