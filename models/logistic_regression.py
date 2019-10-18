import math
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import datareader_csv
from models.model_utility import split_feature_label, scale_features

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)

    #Load data
    os.chdir("..")
    training = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_37582_100ms.csv", 0)
    test = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_9396_100ms.csv", 0)

    #Split it into training and test data and scale it
    X_train, y_train = split_feature_label(training)
    X_test, y_test = split_feature_label(test)
    X_train, X_test = scale_features(X_train, X_test)

    #create Logistic Regression model and fit it
    LR = LogisticRegression()
    LR.fit(X_train, y_train)

    #Test accuracy
    y_predict = LR.predict(X_test)
    print(classification_report(y_test, y_predict))

def logistic_regression(X_train, y_train):
    # Define the model
    print("the Logistic Regression model is now being created and fitted with data")
    logistic_regr = LogisticRegression()

    #Fit the model
    logistic_regr.fit(X_train, y_train)

    #model get returned so accuracy can get calculated
    print("The Logistic Regression Model has been created, prediction of accuracy is now being calculated")
    return logistic_regr