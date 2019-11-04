from sklearn.neural_network import MLPClassifier


def mlp(parameters):  # baseline parameters
    return MLPClassifier(hidden_layer_sizes=(12, 2), max_iter=500).set_params(**parameters)