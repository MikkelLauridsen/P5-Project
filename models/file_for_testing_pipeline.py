import os
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from models.model_utility import split_feature_label

import datareader_csv

#DOESNT WORK
os.chdir("..")
training = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_77441_100ms.csv", 0)
test = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_19361_100ms.csv", 0)
print(pd.training.dtypes)
X_train, y_train = split_feature_label(training)
X_test, y_test = split_feature_label(test)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_features = training.select_dtypes(include=['int64', 'float64','object']).columns
categorical_features = training.select_dtypes(include=['object'], axis=1).columns

rf = Pipeline(steps=[('preprocessor', 'preprocessor'),('classifier',KNeighborsClassifier)])
print (rf)