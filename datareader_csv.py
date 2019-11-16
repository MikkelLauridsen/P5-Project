"""Functions for loading and parsing data from the csv files into lists of Messages."""
import csv
import pandas as pd
import datapoint
import message
import os
from metrics import Metrics, Result, get_metrics_path
from datapoint import datapoint_features


def __load_data(filepath, parse_func, start, limit, verbose=False):
    # Loading data from the file row by row and parsing the rows using the specified parse function.

    print(f"Started reading from {filepath}")

    data = []

    try:
        # Creating a dataframe that contains the specified rows of the specified csv file.
        df = pd.read_csv(filepath, header=0, skiprows=range(1, start + 1), nrows=limit)
    except FileNotFoundError:
        print("The file does not exist")
        return

    for count, row in enumerate(df.values.tolist()):
        if not verbose and count % 50000 == 0:
            print(f"Reading message: {str(count)}")

        data.append(parse_func(row))

    return data


def load_messages(filepath, start=0, limit=None, verbose=False):
    """
    Loading a specified amount of messages from the filepath.

    :param filepath: The filepath where the data should be read from.
    :param start: The index indicating where the function should start loading messages.
    :param limit: The index indicating where the function should stop loading messages.
    :param verbose: Boolean indicator controlling how much information about the loading process that should be printed.
    :return: A list of Message objects corresponding to the given parameters.
    """
    return __load_data(filepath, message.parse_csv_row, start, limit, verbose)


def load_datapoints(filepath, start=0, limit=None, verbose=False):
    """
    Loading a specified amount of data points from the filepath.

    :param filepath: The filepath where the data should be read from.
    :param start: The index indicating where the function should start loading data points.
    :param limit: The index indicating where the function should stop loading data points.
    :param verbose: Boolean indicator controlling how much information about the loading process that should be printed.
    :return: A list of DataPoint objects corresponding to the given parameters.
    """
    # Check if csv format matches DataPoint structure
    df = pd.read_csv(filepath, header=0, nrows=0)
    matching_header, diff = datapoint.is_header_matching(df.columns)
    if not matching_header:
        print("Found mismatching datapoint and csv file structure")
        print(f"datapoint: {list(datapoint.datapoint_attributes)}")
        print(f"csv      : {list(df.columns)}")
        print(f"diff     : {diff}")

    return __load_data(filepath, datapoint.parse_csv_row, start, limit, verbose)


def load_feature_durations(filepath):
    """
    Loads the average time of feature calculation from the specified filepath.

    :param filepath: The filepath where the data should be read from.
    :return: A dict containing the average times it took to calculate features.
    """
    feature_durations = {}

    with open(filepath, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader, None)

        for row in reader:
            feature_durations[row[0]] = float(row[1])

    return feature_durations


def load_attack_free1(start=0, limit=None, verbose=False):
    """Loads data from "Attack_free_dataset.csv"."""
    return load_messages("data/manipulated/Attack_free_dataset.csv", start, limit, verbose)


def load_attack_free2(start=0, limit=None, verbose=False):
    """Loads data from "Attack_free_dataset2.csv"."""
    return load_messages("data/csv/Attack_free_dataset2.csv", start, limit, verbose)


def load_impersonation_1(start=0, limit=None, verbose=False):
    """Loads data from "Impersonation_attack_dataset.csv"."""
    return load_messages("data/csv/Impersonation_attack_dataset.csv", start, limit, verbose)


def load_impersonation_2(start=0, limit=None, verbose=False):
    """Loads data from "170907_impersonation.csv"."""
    return load_messages("data/manipulated/170907_impersonation.csv", start, limit, verbose)


def load_impersonation_3(start=0, limit=None, verbose=False):
    """Loads data from "170907_impersonation_2.csv"."""
    return load_messages("data/manipulated/170907_impersonation_2.csv", start, limit, verbose)


def load_dos(start=0, limit=None, verbose=False):
    """Loads data from "DoS_attack_dataset.csv"."""
    return load_messages("data/manipulated/DoS_attack_dataset.csv", start, limit, verbose)


def load_modified_dos(start=0, limit=None, verbose=False):
    """Loads data from "DoS_manipulated.csv"."""
    return load_messages("data/manipulated/DoS_manipulated.csv", start, limit, verbose)


def load_fuzzy(start=0, limit=None, verbose=False):
    """Loads data from "Fuzzy_attack_dataset.csv"."""
    return load_messages("data/manipulated/Fuzzy_attack_dataset.csv", start, limit, verbose)


def load_metrics(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset, is_test=False):
    """Loads metrics from the file associated with the specified parameters.

    :param period_ms: the used window size (int ms).
    :param stride_ms: the used step-size (int ms).
    :param imp_split: a flag indicating whether the impersonation labels were split.
    :param dos_type: a string indicating the type of DoS dataset used ('modified', 'original').
    :param model: a string indicating the model used ('mlp', 'knn', 'svm', 'rf', 'nbc', 'lr', 'dt', 'bn').
    :param baseline: a flag indicating whether baseline parameters were used.
    :param subset: a list of feature labels, corresponding to the features used.
    :param is_test: a flag indicating whether the test set was used.
    :return: a dictionary of Metrics objects, with a key for each class as well as 'total'.
    """

    path, _ = get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset, is_test=is_test)
    metrics = {}

    with open(path, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        # Skip header
        next(reader, None)

        # For each row in the file, construct a Metrics object
        for row in reader:
            metrics[row[0]] = Metrics(*[float(string) for string in row[1:]])

    return metrics


def load_times(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset, is_test=False):
    """Loads time scores from the file associated with the specified parameters.

    :param period_ms: the used window size (int ms).
    :param stride_ms: the used step-size (int ms).
    :param imp_split: a flag indicating whether the impersonation labels were split.
    :param dos_type: a string indicating the type of DoS dataset used ('modified', 'original').
    :param model: a string indicating the model used ('mlp', 'knn', 'svm', 'rf', 'nbc', 'lr', 'dt', 'bn').
    :param baseline: a flag indicating whether baseline parameters were used.
    :param subset: a list of feature labels, corresponding to the features used.
    :param is_test: a flag indicating whether the test set was used.
    :return: a dictionary of times, with keys 'model_time', 'feature_time', 'total_time'.
    """
    path, _ = get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset, True, is_test)

    with open(path, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        # Skip header
        next(reader, None)

        row = next(reader, None)

    return {'model_time': float(row[0]), 'feature_time': float(row[1]), 'total_time': float(row[2])}


def __load_result(path):
    # Loads results from the file at the specified path and returns a corresponding Result object

    labels = datapoint_features  # List of all feature labels

    # Split path on '\' and find the index of 'result' in the resulting list
    substrings = (path[:-4]).split("\\")
    begin = len(substrings) - 1

    while substrings[begin] != 'result':
        begin -= 1

    # Extract information about the used dataset, situated at specific offsets from the 'begin' index
    baseline = substrings[begin + 1] == 'baseline'
    model = substrings[begin + 2]
    imp_split = substrings[begin + 3] != 'imp_full'
    dos_type = substrings[begin + 4]

    # Split the remaining string on '_' and discard 'mixed'
    file_split = substrings[begin + 5].split("_")[1:]

    is_test = file_split[0] == 'test'
    period_ms = int((file_split[2])[:-2])
    stride_ms = int((file_split[3])[:-2])
    subset = []

    # Use each substring from index 4 as an index into the list of all feature labels, to establish a feature subset
    for substring in file_split[4:]:
        index = int(substring)

        subset.append(labels[index])

    # Use the gathered information to load the metrics and times
    metrics = load_metrics(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset)
    times = load_times(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset)

    return Result(period_ms, stride_ms, model, imp_split, dos_type, baseline, subset, is_test, metrics, times)


def __load_results(directory):
    # Recursively traverses the specified directory and returns a list of Result objects, one for each file found
    results = []

    for path in os.listdir(directory):
        abs_path = os.path.join(directory, path)

        if os.path.isfile(abs_path):
            if __is_score_file(abs_path):
                results.append(__load_result(abs_path))
        else:
            results += __load_results(abs_path)

    return results


def load_all_results():
    """Returns a list of Result objects, representing all result files in the 'result' folder."""
    directory = os.getcwd() + "\\result"

    return __load_results(directory)


def __is_score_file(path):
    # Returns whether the specified path points to a score file
    substrings = (path[:-4]).split("\\")
    substring = substrings[len(substrings) - 1]

    return substring.split("_")[2] == "score"
