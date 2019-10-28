from sklearn.neighbors import KNeighborsClassifier


def knn(X_train, y_train, parameters={}):
    return KNeighborsClassifier(**parameters).fit(X_train, y_train)


# def knn(X_train, y_train, length):
#    # Calculate k
#    print("The knn model is now being created and fitted with data")
#    k = round(math.sqrt(length), 0)
#    if k % 2 == 0:
#        k = k - 1
#    k = int(k)
#
#    # Define the model
#    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean', metric_params=None,
#                               n_jobs=-1, n_neighbors=k, p=2, weights='distance')
#
#    #Fit the model
#    knn.fit(X_train, y_train)
#
#    # Model send back for accuracy
#    print("Knn model has been created, prediction of accuracy is now being calculated")
#    return knn
