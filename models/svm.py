from sklearn.svm import SVC
import models.model_utility
from sklearn.model_selection import GridSearchCV


def svm(X_train, y_train):
    # set up the parameter grid
    print("Grid for SVM is now being set up")
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 50, 100], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 50, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', "poly", "sigmoid"]},
    ]
    print("Grid for SVM has now been set up, hyper parameters are being found")

    # Find hyperparameters
    grid_s = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, scoring="f1_macro")
    grid_s.fit(X_train, y_train)

    print(grid_s.best_estimator_)
    print(grid_s.best_score_)
    print(grid_s.best_params_)


if __name__ == "__main__":
    X_train, y_train = models.model_utility.get_standard_feature_split()

    svm(X_train, y_train)
