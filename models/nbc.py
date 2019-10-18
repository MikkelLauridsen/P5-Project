import math
import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

import datareader_csv
from models.model_utility import split_feature_label, scale_features

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    #Load data
    os.chdir("..")
    training = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_37582_100ms.csv", 0)
    test = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_9396_100ms.csv", 0)

    #split and scale data
    X_train, y_train = split_feature_label(training)
    X_test, y_test = split_feature_label(test)
    X_train, X_test = scale_features(X_train, X_test)

    #create model and fit model
    nbc = GaussianNB()
    nbc.fit(X_train, y_train)

    #test accuracy of the model
    y_predict = nbc.predict(X_test)
    print(classification_report(y_test, y_predict))

def nbc(X_train, y_train):
    # define the model
    print("Naive Bayes model is now being created and fitted with data")
    naive_bayes = GaussianNB()

    #Fit the model
    naive_bayes.fit(X_train, y_train)
    print("The Naive Bayes model has been created and prediction of accuracy is now being calculated")

    #Return the model to get accuracy prediction
    return naive_bayes

