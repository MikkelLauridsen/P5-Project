from sklearn.preprocessing import StandardScaler

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
