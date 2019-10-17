import math

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def panda():
    print("This is a test panda, don't mind the test panda, it's harmless")
    return


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
    knn.fit(X_train, y_train)
    print("Knn model has been created, prediction is now being calculated")
    return knn


def logisticRegression(X_train, y_train):
    # Define the model
    print("the Logistic Regression model is now being created and fitted with data")
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    print("The Logistic Regression Model has been created, prediction is now being calculated")
    return logisticRegr


def nbc(X_train, y_train):
    # define the model
    print("Naive Bayes model is now being created and fitted with data")
    naiveBayes = GaussianNB()
    naiveBayes.fit(X_train, y_train)
    print("The Naive Bayes model has been created and prediction is now being calculated")
    return naiveBayes


def svm(X_train, y_train):
    # set up the parameter grid
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', "poly", "sigmoid"]},
    ]
    return
