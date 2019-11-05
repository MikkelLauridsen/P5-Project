"""Functions for loading and parsing data from the csv files into lists of Messages."""
import csv
import pandas as pd
import datapoint
import message
import os
from glob import glob
from metrics import Metrics, Result, get_metrics_path
from datapoint import datapoint_attributes


def __load_data(filepath, parse_func, start, limit, verbose=False):
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
    return __load_data(filepath, message.parse_csv_row, start, limit, verbose)


def load_datapoints(filepath, start=0, limit=None, verbose=False):
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
    feature_durations = {}

    with open(filepath, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader, None)

        for row in reader:
            feature_durations[row[0]] = float(row[1])

    return feature_durations


# Loads data from "Attack_free_dataset.csv"
def load_attack_free1(start=0, limit=None, verbose=False):
    return load_messages("data/manipulated/Attack_free_dataset.csv", start, limit, verbose)


# Loads data from "Attack_free_dataset2.csv"
def load_attack_free2(start=0, limit=None, verbose=False):
    return load_messages("data/csv/Attack_free_dataset2.csv", start, limit, verbose)


# Loads data from "Impersonation_attack_dataset.csv"
def load_impersonation_1(start=0, limit=None, verbose=False):
    return load_messages("data/csv/Impersonation_attack_dataset.csv", start, limit, verbose)


# Loads data from "170907_impersonation.csv"
def load_impersonation_2(start=0, limit=None, verbose=False):
    return load_messages("data/manipulated/170907_impersonation.csv", start, limit, verbose)


# Loads data from "170907_impersonation_2.csv"
def load_impersonation_3(start=0, limit=None, verbose=False):
    return load_messages("data/manipulated/170907_impersonation_2.csv", start, limit, verbose)


# Loads data from "DoS_attack_dataset.csv"
def load_dos(start=0, limit=None, verbose=False):
    return load_messages("data/manipulated/DoS_attack_dataset.csv", start, limit, verbose)


def load_modified_dos(start=0, limit=None, verbose=False):
    return load_messages("data/manipulated/DoS_manipulated.csv", start, limit, verbose)


# Loads data from "Fuzzy_attack_dataset.csv"
def load_fuzzy(start=0, limit=None, verbose=False):
    return load_messages("data/manipulated/Fuzzy_attack_dataset.csv", start, limit, verbose)


def load_metrics(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset):
    path, _ = get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset)
    metrics = {}

    with open(path, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        # skip header
        next(reader, None)

        for row in reader:
            metrics[row[0]] = Metrics(*[float(string) for string in row[1:]])

    return metrics


def load_times(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset):
    path, _ = get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset, True)

    with open(path, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        # skip header
        next(reader, None)

        row = next(reader, None)

    return {'model_time': row[0], 'feature_time': row[1], 'total_time': row[2]}


def load_result(path):
    labels = list(datapoint_attributes)[2:]
    substrings = (path[:-4]).split("\\")
    begin = len(substrings) - 1

    while substrings[begin] != 'result':
        begin -= 1

    baseline = substrings[begin + 1] == 'baseline'
    model = substrings[begin + 2]
    imp_split = substrings[begin + 3] != 'imp_full'
    dos_type = substrings[begin + 4]

    file_split = substrings[begin + 5].split("_")[2:]

    period_ms = int((file_split[0])[:-2])
    stride_ms = int((file_split[1])[:-2])
    subset = []

    for substring in file_split[2:]:
        index = int(substring)

        subset.append(labels[index])

    metrics = load_metrics(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset)
    times = load_times(period_ms, stride_ms, imp_split, dos_type, model, baseline, subset)

    return Result(period_ms, stride_ms, model, imp_split, dos_type, baseline, subset, metrics, times)


def load_all_metric_time_pairs(directory):
    results = []

    for path in os.listdir(directory):
        abs_path = os.path.join(directory, path)

        if os.path.isfile(abs_path):
            results.append(load_result(abs_path))

    return results
