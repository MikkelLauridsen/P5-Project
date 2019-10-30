from sklearn.neural_network import MLPClassifier


def mlp():  # baseline parameters
    return MLPClassifier(hidden_layer_sizes=(12, 2))


#def mlp(X_train, y_train, parameters={}):
#    # The grid gets setup
#    print("Grid for mlp is now being set up")
#    parameter_space = [{'solver': ["adam"],
#                        'alpha': [1.e-06],
#                        'hidden_layer_sizes': [(16, 4)],
#                        'random_state': [1]}]
#
#    # Find the best hyperparameters
#    print("Grid for mlp has now been set up, hyper parameters are being found")
#    mlp_model = (best_hyper_parameters_for_all_model(MLPClassifier(), parameter_space, X_train, y_train))
#
#    # Model gets send back to have accuracy predicted
#    print("The mlp model has now been created, and prediction of accuracy is now being calculated")
#
#    return mlp_model
