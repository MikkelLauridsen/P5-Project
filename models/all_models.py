import math
from functools import partial

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from models.model_utility import find_best_hyperparameters, find_best_hyperparameters2


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
    print("Knn model has been created, prediction of accuracy is now being calculated")
    return knn


def logistic_regression(X_train, y_train):
    # Define the model
    print("the Logistic Regression model is now being created and fitted with data")
    logistic_regr = LogisticRegression()
    logistic_regr.fit(X_train, y_train)
    print("The Logistic Regression Model has been created, prediction of accuracy is now being calculated")
    return logistic_regr


def nbc(X_train, y_train):
    # define the model
    print("Naive Bayes model is now being created and fitted with data")
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    print("The Naive Bayes model has been created and prediction of accuracy is now being calculated")
    return naive_bayes


def svm(X_train, y_train):
    # set up the parameter grid
    print("Grid for SVM is now being set up")
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', "poly", "sigmoid"]},
    ]
    print("Grid for SVM has now been set up, hyper parameters are being found")
    svm_model = find_best_hyperparameters2(SVC(), param_grid, X_train, y_train)
    print("The SVM model has now been created, and prediction of accuracy is now being calculated")
    return svm_model


def decision_trees(X_train, y_train):
    # set up the parameter grid
    print("Grid for decision trees is now being set up")
    parameter_grid = [{'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       'min_samples_split': [2, 3, 5, 8],
                       'max_features': [None, 2, 3, 5],
                       'criterion': ['gini', 'entropy']}]
    print("Grid for decision trees has now been set up, hyper parameters are being found")
    decision_trees_model = find_best_hyperparameters2(DecisionTreeClassifier(), parameter_grid, X_train, y_train)
    print("The decision tree model has now been created, and prediction of accuracy is now being calculated")
    return decision_trees_model


def ada_boost(X_train, y_train):
    print("Grid for ada boost is now being set up")
    parameter_grid = [{'n_estimators': [3, 5, 8, 9, 10, 11, 12, 13],
                       'algorithm': ['SAMME.R', 'SAMME'],
                       'learning_rate': [0.1, 0.5, 1.0]}]
    print("Grid for ada boost has now been set up, hyper parameters are being found")
    ada_boost_model = find_best_hyperparameters2(AdaBoostClassifier(), parameter_grid, X_train, y_train)
    print("The ada boost model has now been created, and prediction of accuracy is now being calculated")
    return ada_boost_model


def mlp(X_train, y_train):
    print("Grid for mlp is now being set up")
    parameter_space = [{'solver': ["adam"],
                        'alpha': [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06],
                        'hidden_layer_sizes': [(12, 4), (14, 4), (16, 4)],
                        'random_state': [1]}]
    print("Grid for mlp has now been set up, hyper parameters are being found")
    mlp_model = (find_best_hyperparameters2(MLPClassifier(), parameter_space, X_train, y_train))
    print("The mlp model has now been created, and prediction of accuracy is now being calculated")
    return mlp_model


def random_forest(X_train, y_train):
    print("Grid for Random forest is being set up")
    parameter_grid = [{'max_depth': [2, 5, 9],
                       'n_estimators': [3],
                       'min_samples_split': [2, 3, 5],
                       'max_features': [None, 2, 3],
                       'criterion': ['gini'],
                       'bootstrap': [True]}]
    print("Grid for Random forest has been set up, hyper parameters are being found")
    random_forest_model = find_best_hyperparameters2(RandomForestClassifier(), parameter_grid, X_train, y_train)
    print("The Random forest model has now been created, and prediction of accuracy is now being calculated")
    return random_forest_model

def all_models_run(X_train,y_train,training,X_test,y_test):
    cmd = {'1': partial(knn, X_train, y_train, len(training)),
           '2': partial(logistic_regression, X_train, y_train),
           '3': partial(svm, X_train, y_train),
           '4': partial(decision_trees, X_train, y_train),
           '5': partial(ada_boost, X_train, y_train),
           '6': partial(nbc, X_train, y_train),
           '7': partial(mlp, X_train, y_train),
           '8': partial(random_forest, X_train, y_train)}
    print("All models now being calculated, please buckle in")
    for x in range (0,9):
        model=cmd[x]()
        y_predict = model.predict(X_test)
        print(classification_report(y_test, y_predict))
    return model

