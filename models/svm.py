import datareader_csv
import idpoint

import sklearn

idpoints = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_32888_100ms.csv")

# The label list that will contain the injected status for each idpoint.
y = []

# The instance list that will contain all the features for each instance.
X = []

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

print(y[0])
print(X[0])
