import os
from functools import partial

from pip._vendor.distlib.compat import raw_input
from sklearn.metrics import classification_report

import datareader_csv
from models.AllModels import knn, panda, logisticRegression, svm
from models.model_utility import split_feature_label, scale_features

# Creates a list of all the models
differentModels = ["knn", "random forest", "ada boost", "decision trees", "lgr", "mlp", "nbc", "svm"]
# Reads in the data from the original file
os.chdir("..")
training = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_77441_100ms.csv", 0)
test = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_19361_100ms.csv", 0)

# Splits it into training and test, then scales it
X_train, y_train = split_feature_label(training)
X_test, y_test = split_feature_label(test)
X_train, X_test = scale_features(X_train, X_test)

# Directory for the different models
cmd = {'panda': panda,
       'knn': partial(knn, X_train, y_train, len(training)),
       'lgr': partial(logisticRegression, X_train, y_train),
       'svm': partial(svm, X_train, y_train)}

# Asks the user to decide which model it would like to run
print("Which model would you like to use? Options are:")
for x in range(len(differentModels)):
    print(differentModels[x])
response = raw_input().lower()
while response not in differentModels:
    response = raw_input("please type in a correct model")
print(response)
model = cmd[response]()
# predict the model
y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))
