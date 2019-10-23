import pandas as pd
from datasets import load_or_create_datasets
from models.model_utility import scale_features, split_feature_label
from datapoint import datapoint_attributes
import matplotlib.pyplot as plt
import seaborn as sns


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
    datapoints1, datapoints2 = load_or_create_datasets(period_ms=100,
                                                       overlap_ms=100,
                                                       impersonation_split=False,
                                                       dos_type='modified')

    X_1, y_1 = split_feature_label(datapoints1)
    X_2, y_2 = split_feature_label(datapoints2)
    X_1, X_2 = scale_features(X_1, X_2)
    X = list(X_1) + list(X_2)
    y = list(y_1) + list(y_2)

    add_labels(X, y)
    column_labels = list(datapoint_attributes)[2:] + ['normal', 'dos', 'fuzzy', 'impersonation']

    # get correlations of each features in dataset
    data = pd.DataFrame(X, columns=column_labels)
    corrmat = data.corr(method='spearman')
    plt.figure(figsize=(20, 20))

    # plot heat map
    g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
    plt.show()
