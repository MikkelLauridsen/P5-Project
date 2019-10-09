import idpoint
import datareader_csv


def split_feature_label():
    idpoints = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_32888_100ms.csv")

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
