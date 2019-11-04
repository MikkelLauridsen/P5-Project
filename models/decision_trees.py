from sklearn.tree import DecisionTreeClassifier
import models.model_utility
from sklearn.model_selection import GridSearchCV


def dt():  # baseline parameters
    return DecisionTreeClassifier(max_depth=9)


def decision_trees(X_train, y_train):
    # set up the parameter grid
    print("Grid for decision trees is now being set up")
    parameter_grid = [{'max_depth': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233],
                       'min_samples_split': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233],
                       'criterion': ['gini', 'entropy']}]
    print("Grid for decision trees has now been set up, hyper parameters are being found")

    grid_s = GridSearchCV(DecisionTreeClassifier(), parameter_grid, cv=5, scoring="f1_macro", verbose=10)
    grid_s.fit(X_train, y_train)

    print(grid_s.best_params_)
    print(grid_s.best_score_)


if __name__ == "__main__":
    X_train, y_train = models.model_utility.get_standard_feature_split()
    decision_trees(X_train, y_train)
