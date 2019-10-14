import math
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import datareader_csv
from models.model_utility import split_feature_label, scale_features

np.set_printoptions(threshold=sys.maxsize)

os.chdir("..")
training = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_37582_100ms.csv", 0)
test = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_9396_100ms.csv", 0)

X_train, y_train = split_feature_label(training)
X_test, y_test = split_feature_label(test)
X_train, X_test = scale_features(X_train, X_test)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_predict = logisticRegr.predict(X_test)
print(classification_report(y_test, y_predict))