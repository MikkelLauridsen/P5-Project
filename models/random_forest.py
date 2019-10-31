from sklearn.ensemble import RandomForestClassifier


def rf():  # baseline parameters
    return RandomForestClassifier(n_estimators=5, max_depth=9)


# Creates the random forest model
#def random_forest(X_train, y_train):
#    # Sets up the parameter grid for random forest
#    print("Grid for Random forest is being set up")
#    parameter_grid = [{'max_depth': [2, 5, 9],
#                       'n_estimators': [3],
#                       'min_samples_split': [2, 3, 5],
#                       'max_features': [None, 2, 3],
#                       'criterion': ['gini'],
#                       'bootstrap': [True]}]
#    print("Grid for Random forest has been set up, hyper parameters are being found")
#
#    # Finds the hyper parameters
#    random_forest_model = best_hyper_parameters_for_all_model(RandomForestClassifier(), parameter_grid, X_train, y_train)
#    print("The Random forest model has now been created, and prediction of accuracy is now being calculated")
#
#    # Returns the model to use prediction on
#    return random_forest_model
