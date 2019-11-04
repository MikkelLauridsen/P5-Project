from sklearn.naive_bayes import GaussianNB


def nbc(parameters):  # baseline parameters
    return GaussianNB().set_params(**parameters)
