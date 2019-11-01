from sklearn.neighbors import KNeighborsClassifier
import models.model_utility
from sklearn.model_selection import GridSearchCV


def knn():  # baseline parameters
    return KNeighborsClassifier()


def knn_hyperparameter(X_train, y_train):
    """Finding the best hyperparameters for knn based on given training data."""
    # set up the parameter grid
    param_grid = [
        {
            "n_neighbors": [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    ]

    # Find hyperparameters
    grid_s = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=10)
    grid_s.fit(X_train, y_train)

    print(grid_s.best_estimator_)
    print(grid_s.best_score_)
    print(grid_s.best_params_)


if __name__ == "__main__":
    X_train, y_train = models.model_utility.get_standard_feature_split()

    knn_hyperparameter(X_train, y_train)
