import datareader_csv
import idpoint
import models.model_utility

import sklearn

training_filepath = "data/idpoint_dataset/mixed_training_135520_100ms.csv"
test_filepath = "data/idpoint_dataset/mixed_test_29041_100ms.csv"

X_train, y_train = models.model_utility.split_feature_label(datareader_csv.load_idpoints(training_filepath))
X_test, y_test = models.model_utility.split_feature_label(datareader_csv.load_idpoints(test_filepath))

X_train, X_test = models.model_utility.scale_features(X_train, X_test)

print(y_train[0])
print(X_train[0])

print(y_test[0])
print(X_test[0])
