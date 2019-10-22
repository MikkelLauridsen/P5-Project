import pandas as pd
import numpy as np
import sys
from id_based_datasets import load_or_create_datasets
from models.model_utility import scale_features, split_feature_label
from idpoint import idpoint_attributes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier

def add_labels(X, y):
    for i in range(len(X)):
        normal = dos = fuzzy = impersonation = 0.0

        if y[i] == 'normal':
            normal = 1.0
        elif y[i] == 'dos':
            dos = 1.0
        elif y[i] == 'fuzzy':
            fuzzy = 1.0
        else:
            impersonation = 1.0

        X[i] = list(X[i]) + [normal, dos, fuzzy, impersonation]


if __name__ == '__main__':
    idpoints1, idpoints2 = load_or_create_datasets(period_ms=10, overlap_ms=10, impersonation_split=False, dos_type='modified')
    X_1, y_1 = split_feature_label(idpoints1)
    X_2, y_2 = split_feature_label(idpoints2)
    X_1, X_2 = scale_features(X_1, X_2)
    X = list(X_1) + list(X_2)
    y = list(y_1) + list(y_2)

    add_labels(X, y)
    column_labels = list(idpoint_attributes)[2:] + ['normal', 'dos', 'fuzzy', 'impersonation']

    # get correlations of each features in dataset
    data = pd.DataFrame(X, columns=column_labels)
    corrmat = data.corr(method='spearman')
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))

    # plot heat map
    g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()


# old feature selection below

#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#
#Read in the data
#data = pd.read_csv("data/idpoint_dataset/mixed_training_37582_100ms.csv")
#
#select the right columns
#X = data.iloc[:,0:20]
#y = data.iloc[:,1]
#
#get correlations of each features in dataset
#corrmat = data.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
#
#plot heat map
#g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.ensemble import ExtraTreesClassifier
#
#Read in the data
#data = pd.read_csv("data/idpoint_dataset/mixed_training_37582_100ms.csv")
#
#select the right columns
#X = data.iloc[:,2:19]
#y = data.iloc[:,1]
#
#create and fit the model
#model = ExtraTreesClassifier()
#model.fit(X,y)
#print(model.feature_importances_)
#
#plot graph of feature importances for better visualization
#important_features = pd.Series(model.feature_importances_, index=X.columns)
#important_features.nlargest(19).plot(kind='barh')
#plt.show()
