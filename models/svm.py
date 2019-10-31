from sklearn.svm import SVC


def svm():
    return SVC(max_iter=1000)


#def svm(X_train, y_train):
#    # set up the parameter grid
#    print("Grid for SVM is now being set up")
#    param_grid = [
#        {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
#        {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', "poly", "sigmoid"]},
#    ]
#    print("Grid for SVM has now been set up, hyper parameters are being found")
#
#    #Find hyperparameters
#    svm_model = best_hyper_parameters_for_all_model(SVC(), param_grid, X_train, y_train)
#
#    # Model gets send back to get accuracy taken
#    print("The SVM model has now been created, and prediction of accuracy is now being calculated")
#    return svm_model