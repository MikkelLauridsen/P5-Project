import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import idpoint


# Splitting the training data into feature and label lists.
def split_feature_label(idpoints):
    # The instance list that will contain all the features for each instance.
    X = []
    # The label list that will contain the injected status for each idpoint.
    y = []

    # Going through each instance and extracting the features and labels.
    for instance in idpoints:
        features = []
        for attr in idpoint.idpoint_attributes:
            if attr == "time_ms":
                pass
            elif attr == "is_injected":
                y.append(instance.is_injected)
            else:
                features.append(getattr(instance, attr))

        X.append(features)

    return X, y


# Fitting a transformation on the training features and scaling both the training features and test features to it.
def scale_features(X_training, X_test):
    scaler = StandardScaler()

    scaler.fit(X_training)

    # Transforming the training and test features with the fitted scaler.
    X_training = scaler.transform(X_training)
    X_test = scaler.transform(X_test)

    return X_training, X_test


# Finds the best combination of hyperparameters and prints the results.
def find_best_hyperparameters(estimator, parameter_grid, X_train, y_train, X_test, y_test):
    # Creating the grid search that uses the parameter grid to find the best hyperparameters.
    grid_s = GridSearchCV(estimator, parameter_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)
    grid_s.fit(X_train, y_train)

    print(f"parameters found: {grid_s.best_params_}")

    y_predict = grid_s.predict(X_test)

    print(classification_report(y_test, y_predict))

    return y_predict