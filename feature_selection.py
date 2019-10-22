import pandas as pd
import numpy as np
import sys
from id_based_datasets import load_or_create_datasets
from models.model_utility import scale_features, split_feature_label
from idpoint import idpoint_attributes
import matplotlib.pyplot as plt


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
    idpoints1, idpoints2 = load_or_create_datasets(period_ms=10, overlap_ms=10, impersonation_split=False, dos_type='modified')
    X_1, y_1 = split_feature_label(idpoints1)
    X_2, y_2 = split_feature_label(idpoints2)
    X_1, X_2 = scale_features(X_1, X_2)
    X = list(X_1) + list(X_2)
    y = list(y_1) + list(y_2)

    add_labels(X, y)
    column_labels = list(idpoint_attributes)[2:] + ['normal', 'dos', 'fuzzy', 'impersonation']
    plt.matshow(pd.DataFrame(X, columns=column_labels).corr(method='spearman'))
    plt.colorbar()
    plt.show()