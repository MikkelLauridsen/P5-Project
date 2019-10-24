import pandas as pd
from datasets import load_or_create_datasets
from models.model_utility import scale_features, split_feature_label
from datapoint import datapoint_attributes
import matplotlib.pyplot as plt


# Adds target features in y to corresponding samples in X.
# X and y must be lists.
def add_labels(X, y):
    for i in range(len(X)):
        normal = dos = fuzzy = impersonation = 0.0

        if y[i] == 'normal':
            normal = 1.0
        elif y[i] == 'dos':
            dos = 1.0
        elif y[i] == 'fuzzy':
            fuzzy = 1.0
        else:
            impersonation = 1.0

        X[i] = list(X[i]) + [normal, dos, fuzzy, impersonation]


if __name__ == '__main__':
    period_ms = 10
    overlap_ms = 10

    datapoints1, datapoints2 = load_or_create_datasets(period_ms=period_ms,
                                                       overlap_ms=overlap_ms,
                                                       impersonation_split=False,
                                                       dos_type='modified')

    X_1, y = split_feature_label(datapoints1)
    X_2, _ = split_feature_label(datapoints2)
    X, _ = scale_features(X_1, X_2)  # throw away test data after scaling
    X = list(X)  # convert X and y to lists
    y = list(y)

    add_labels(X, y)
    column_labels = list(datapoint_attributes)[2:] + ['normal', 'dos', 'fuzzy', 'impersonation']

    # get correlations of each feature in dataset
    data = pd.DataFrame(X, columns=column_labels)
    corrmat = data.corr(method='spearman')
    figure = plt.figure(figsize=(22, 15))
    plt.matshow(corrmat, fignum=figure.number)
    plt.colorbar()
    plt.title(f"Correlations at {period_ms}ms windows and {overlap_ms}ms overlap", fontsize=20)
    plt.xticks(range(data.shape[1]), list(range(22)), fontsize=14)
    # add feature names as y-axis labels
    plt.yticks([-0.5] + list(range(data.shape[1])) + [data.shape[1] - 0.5], [""] + column_labels + [""], fontsize=14)
    plt.show()

    print("relevant features being calculated")
    target = ["normal", "dos", "impersonation", "fuzzy"]

    for x in target:
        cor_target = abs(corrmat[x])
        relevant_features = cor_target[cor_target > 0.3]
        print(x)
        print(relevant_features)
