import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read in the data
data = pd.read_csv("data/idpoint_dataset/mixed_training_37582_100ms.csv")

#select the right columns
X = data.iloc[:,0:20]
y = data.iloc[:,1]

#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

#Read in the data
data = pd.read_csv("data/idpoint_dataset/mixed_training_37582_100ms.csv")

#select the right columns
X = data.iloc[:,2:19]
y = data.iloc[:,1]

#create and fit the model
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)

#plot graph of feature importances for better visualization
important_features = pd.Series(model.feature_importances_, index=X.columns)
important_features.nlargest(19).plot(kind='barh')
plt.show()
