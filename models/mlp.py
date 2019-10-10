from models.model_utility import split_feature_label
from models.model_utility import scale_features
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import datareader_csv

# load training idpoints
training_points = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_135520_100ms.csv", 0)
test_points = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_29041_100ms.csv", 0)

X_train, y_train = split_feature_label(training_points)
X_test, y_test = split_feature_label(test_points)
X_train, X_test = scale_features(X_train, X_test)

parameter_space = [{'solver': ['lbfgs'],
                    'alpha': [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06],
                    'hidden_layer_sizes': [(16, 2), (12, 3)],
                    'random_state': [1]}]

grid_s = GridSearchCV(MLPClassifier(), parameter_space, cv=5, n_jobs=-1, scoring='accuracy', verbose=10)
grid_s.fit(X_train, y_train)

print(f"parameters found: {grid_s.best_params_}")

y_predict = grid_s.predict(X_test)

print(classification_report(y_test, y_predict))



