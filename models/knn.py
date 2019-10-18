import math
import os
import datareader_csv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from models.model_utility import split_feature_label, scale_features

if __name__ == "__main__":
    os.chdir("..")
    training = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_37582_100ms.csv", 0)
    test = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_9396_100ms.csv", 0)
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



def knn(X_train, y_train, length):
    # Calculate k
    print("The knn model is now being created and fitted with data")
    k = round(math.sqrt(length), 0)
    if k % 2 == 0:
        k = k - 1
    k = int(k)

    # Define the model
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean', metric_params=None,
                               n_jobs=-1, n_neighbors=k, p=2, weights='distance')

    #Fit the model
    knn.fit(X_train, y_train)

    # Model send back for accuracy
    print("Knn model has been created, prediction of accuracy is now being calculated")
    return knn
