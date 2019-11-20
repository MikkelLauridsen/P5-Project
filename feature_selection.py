import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from datasets import load_or_create_datasets
from models.model_utility import scale_features, split_feature_label
from datapoint import datapoint_features
import matplotlib.pyplot as plt


if __name__ == '__main__':
    period_ms = 100
    stride_ms = 100

    datapoints1, datapoints2, _ = load_or_create_datasets(imp_split=True, dos_type='original', force_create=True)

    X_1, y = split_feature_label(datapoints1)
    X_2, _ = split_feature_label(datapoints2)
    X, _ = scale_features(X_1, X_2)  # Throw away validation data after scaling
    X = list(X)  # Convert X and y to lists
    y = list(y)

    column_labels = datapoint_features
    column_labels = [f"{label} {column_labels.index(label)}" for label in column_labels]

    # Get correlations of each feature in dataset
    data = pd.DataFrame(X, columns=column_labels)
    corrmat = data.corr(method='spearman')
    figure = plt.figure(figsize=(22, 15))
    plt.matshow(corrmat, fignum=figure.number)
    plt.colorbar()
    plt.title(f"Correlations at {period_ms}ms windows and {stride_ms}ms overlap", fontsize=20)
    plt.xticks(range(data.shape[1]), list(range(22)), fontsize=14)
    # Add feature names as y-axis labels
    plt.yticks([-0.5] + list(range(data.shape[1])) + [data.shape[1] - 0.5], [""] + column_labels + [""], fontsize=14)
    plt.show()

    # Feature importance
    classifier = ExtraTreesClassifier()
    classifier.fit(X, y)
    importance = pd.Series(classifier.feature_importances_, index=datapoint_features)
    importance.nlargest(15).plot(kind='barh')
    plt.show()
