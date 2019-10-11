import math
import os
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import datareader_csv
from models.model_utility import split_feature_label, scale_features

np.set_printoptions(threshold=sys.maxsize)

os.chdir("..")

# training = pd.read_csv('data/idpoint_dataset/mixed_training_77441_100ms.csv')
# test=pd.read_csv('data/idpoint_dataset/mixed_test_19361_100ms.csv')

training = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_77441_100ms.csv", 0)
test = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_19361_100ms.csv", 0)

X_train, y_train = split_feature_label(training)
X_test, y_test = split_feature_label(test)
X_train, X_test = scale_features(X_train, X_test)

# Calculate n
n = round(math.sqrt(len(training)), 0)
if n % 2 == 0:
    n = n-1
n = int(n)

# Define model
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean', metric_params=None,
                           n_jobs=-1, n_neighbors=n, p=2, weights='distance')
knn.fit(X_train, y_train)
print(knn)

# Predict
y_predict = knn.predict(X_test)

print(classification_report(y_test, y_predict))
