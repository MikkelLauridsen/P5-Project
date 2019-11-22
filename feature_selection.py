import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

import datareader_csv
from datasets import load_or_create_datasets
from models.model_utility import scale_features, split_feature_label
from datapoint import datapoint_features, datapoint_attribute_descriptions
import matplotlib.pyplot as plt

from plotting import feature_plotting

if __name__ == '__main__':
    period_ms = 100
    stride_ms = 100

    dos_type = 'original'  # 'modified' or 'original'
    imp_type = 'imp_split'  # 'imp_split' or 'imp_full'

    datapoints1, datapoints2, _ = load_or_create_datasets(imp_split=imp_type == 'imp_split', dos_type=dos_type, force_create=True)

    X_1, y = split_feature_label(datapoints1)
    X_2, _ = split_feature_label(datapoints2)
    X, _ = scale_features(X_1, X_2)  # Throw away validation data after scaling
    X = list(X)  # Convert X and y to lists
    y = list(y)

    column_labels = datapoint_features
    column_labels = [f"{datapoint_attribute_descriptions[label]} {column_labels.index(label)}" for label in column_labels]

    # Get correlations of each feature in dataset
    data = pd.DataFrame(X, columns=column_labels)
    corrmat = data.corr(method='spearman')
    figure = plt.figure(figsize=(22, 15))
    plt.matshow(corrmat, fignum=figure.number)
    plt.colorbar().ax.tick_params(labelsize=20, length=10)
    # plt.title(f"Correlations at {period_ms}ms windows and {stride_ms}ms overlap", fontsize=30)
    plt.xticks(range(data.shape[1]), list(range(22)), fontsize=20)
    plt.tick_params(length=10, bottom=False)
    plt.clim(-1, 1)
    # Add feature names as y-axis labels
    plt.yticks([-0.5] + list(range(data.shape[1])) + [data.shape[1] - 0.5], [""] + column_labels + [""], fontsize=25)
    plt.savefig('heatmap.png', bbox_inches='tight')
    plt.show()

    # Feature importance
    indices = [datapoint_attribute_descriptions[label] for label in datapoint_features]
    classifier = ExtraTreesClassifier(n_estimators=250)
    classifier.fit(X, y)
    importance = pd.Series(classifier.feature_importances_, index=indices)
    importance.nlargest(15).plot(kind='barh')
    plt.show()

    # Feature durations
    durations_path = f"data\\feature\\{imp_type}\\{dos_type}\\mixed_validation_time_100ms_100ms.csv"
    feature_times = datareader_csv.load_feature_durations(durations_path)
    del feature_times['time_ms']
    del feature_times['class_label']
    feature_plotting.plot_feature_barcharts(feature_times)
