from functools import partial
from sklearn.metrics import classification_report

from models.decision_trees import decision_trees
from models.knn import knn
from models.logistic_regression import logistic_regression
from models.mlp import mlp
from models.nbc import nbc
from models.random_forest import random_forest
from models.svm import svm


# Runs all the models
def all_models_run(X_train, y_train, length, X_test, y_test):
    # creates a dictionary for the different models
    cmd = {1: partial(knn, X_train, y_train, length),
           2: partial(logistic_regression, X_train, y_train),
           3: partial(svm, X_train, y_train),
           4: partial(decision_trees, X_train, y_train),
           6: partial(nbc, X_train, y_train),
           7: partial(mlp, X_train, y_train),
           8: partial(random_forest, X_train, y_train)}
    print("All models now being calculated, please wait")
    # Runs a for loop for all the different models + prediction
    for x in range(1, 9):
        model = cmd[x]()
        y_predict = model.predict(X_test)
        print(classification_report(y_test, y_predict))

    return model
