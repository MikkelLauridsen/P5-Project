import datareader_csv
import idpoint
import models.model_utility

import sklearn


X, y = models.model_utility.split_feature_label()

print(y[0])
print(X[0])
