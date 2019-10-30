from sklearn.naive_bayes import GaussianNB


def nbc():  # baseline parameters
    return GaussianNB()


#def nbc(X_train, y_train):
#    # define the model
#    print("Naive Bayes model is now being created and fitted with data")
#    naive_bayes = GaussianNB()
#
#    # Fit the model
#    naive_bayes.fit(X_train, y_train)
#    print("The Naive Bayes model has been created and prediction of accuracy is now being calculated")
#
#    # Return the model to get accuracy prediction
#    return naive_bayes

