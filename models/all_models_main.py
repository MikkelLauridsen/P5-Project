import os
import datareader_csv
from functools import partial
from pip._vendor.distlib.compat import raw_input
from sklearn.metrics import classification_report
from models.mlp import mlp
from models.all_models import all_models_run
from models.decision_trees import decision_trees
from models.knn import knn
from models.logistic_regression import logistic_regression
from models.nbc import nbc
from models.random_forest import random_forest
from models.svm import svm
from models.model_utility import split_feature_label, scale_features

# Creates a list of all the models
differentModels = ["knn", "random forest", "decision trees", "logistic regression", "mlp", "nbc", "svm", "all"]

# Reads in the data from the original file
os.chdir("..")
training = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_77441_100ms.csv", 0)
test = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_19361_100ms.csv", 0)
# Splits it into training and test, then scales it
X_train, y_train = split_feature_label(training)
X_test, y_test = split_feature_label(test)
X_train, X_test = scale_features(X_train, X_test)
# Directory for the different models
cmd = {'knn': partial(knn, X_train, y_train, len(training)),
       'logistic regression': partial(logistic_regression, X_train, y_train),
       'svm': partial(svm, X_train, y_train),
       'decision trees': partial(decision_trees, X_train, y_train),
       'nbc': partial(nbc, X_train, y_train),
       'mlp': partial(mlp, X_train, y_train),
       'random forest': partial(random_forest, X_train, y_train),
       'all': partial(all_models_run,X_train,y_train, len(training),X_test,y_test)}

# Asks the user to decide which model it would like to run, then sends that input to another function
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
