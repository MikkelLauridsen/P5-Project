from unittest import TestCase
import os


class TestTxtToCsv(TestCase):
    def setUp(self) -> None:
        os.chdir("..")
        messages = []

        # Setting up a test txt file containing the first 10 rows in the file "Attack_free_dataset.txt".
        with open("data/raw/Attack_free_dataset.txt", "r") as reader:
            for i in range(10):
                messages.append(reader.readline())

            with open("tests/attack_free_1_test.txt", "w") as writer:
                for message in messages:
                    writer.write(message)

        messages = []

        # Setting up a test txt file containing the first 10 rows in the file "Attack_free_dataset2.txt".
        with open("data/raw/Attack_free_dataset2.txt", "r") as reader:
            reader.readline()
            for i in range(10):
                messages.append(reader.readline())

            with open("tests/attack_free_2_test.txt", "w") as writer:
                for message in messages:
                    writer.write(message)

    def tearDown(self) -> None:
        # Deleting the files used in the tests.
        delete_list = ["tests/attack_free_1_test.txt", "tests/attack_free_2_test.txt",
                       "tests/attack_free_1_test.csv", "tests/attack_free_2_test.csv"]

        for filepath in delete_list:
            if os.path.exists(filepath):
                os.remove(filepath)
