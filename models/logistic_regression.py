from sklearn.linear_model import LogisticRegression


def lr(X_train, y_train, parameters={}):
    return LogisticRegression(**parameters).fit(X_train, y_train)


#def logistic_regression(X_train, y_train):
#    # Define the model
#    print("the Logistic Regression model is now being created and fitted with data")
#    logistic_regr = LogisticRegression()
#
#    # Fit the model
#    logistic_regr.fit(X_train, y_train)
#
#    # Model get returned so accuracy can get calculated
#    print("The Logistic Regression Model has been created, prediction of accuracy is now being calculated")
#    return logistic_regr
