from sklearn.tree import DecisionTreeClassifier


def dt():  # baseline parameters
    return DecisionTreeClassifier(max_depth=5)


#def decision_trees(X_train, y_train):
#    # set up the parameter grid
#    print("Grid for decision trees is now being set up")
#    parameter_grid = [{'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#                       'min_samples_split': [2, 3, 5, 8],
#                       'max_features': [None, 2, 3, 5],
#                       'criterion': ['gini', 'entropy']}]
#    print("Grid for decision trees has now been set up, hyper parameters are being found")
#
#    # Find hyper parameters
#    decision_trees_model = best_hyper_parameters_for_all_model(DecisionTreeClassifier(), parameter_grid, X_train, y_train)
#
#    # returns the model to calculate the accuracy
#    print("The decision tree model has now been created, and prediction of accuracy is now being calculated")
#    return decision_trees_model
