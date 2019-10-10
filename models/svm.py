import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import datareader_csv
from models.model_utility import scale_features
from models.model_utility import split_feature_label

# Going up one directory so we have access to the below specified files.
os.chdir("..")

training_filepath = "data/idpoint_dataset/mixed_training_135520_100ms.csv"
test_filepath = "data/idpoint_dataset/mixed_test_29041_100ms.csv"

X_train, y_train = split_feature_label(datareader_csv.load_idpoints(training_filepath))
X_test, y_test = split_feature_label(datareader_csv.load_idpoints(test_filepath))

X_train, X_test = scale_features(X_train, X_test)

# Setting up a parameter grid to define the different hyperparameters and their ranges.
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

# Creating the grid search that uses the parameter grid to find the best hyperparameters.
grid_s = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=10)
grid_s.fit(X_train, y_train)

print(f"parameters found: {grid_s.best_params_}")

y_predict = grid_s.predict(X_test)

print(classification_report(y_test, y_predict))