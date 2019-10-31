import csv
import os


def save_feature_durations(feature_durations, path, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Feature', 'Time'])

        for feature in feature_durations.keys():
            writer.writerow([feature, feature_durations[feature]])
