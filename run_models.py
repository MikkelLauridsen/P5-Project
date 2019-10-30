from models.mlp import mlp
from datasets import load_or_create_datasets
from models.model_utility import scale_features, split_feature_label, get_metrics, print_metrics


def select_features(X, feature_index):
    X_sub = []

    for x in X:
        fs = []

        for i in range(len(x)):
            if i == feature_index:
                fs.append(x[i])

        X_sub.append(fs)

    return X_sub


if __name__ == "__main__":

    res = []

    for window_ms in [10, 20, 30, 50, 80, 130]:
        for stride_ms in [10, 20, 30, 50, 80, 130]:
            training_points, test_points = load_or_create_datasets(period_ms=window_ms,
                                                                   shuffle=True,
                                                                   impersonation_split=False,
                                                                   stride_ms=stride_ms,
                                                                   dos_type='modified')

            X_train, y_train = split_feature_label(training_points)
            X_test, y_test = split_feature_label(test_points)
            X_train, X_test = scale_features(X_train, X_test)

            print(f"Generated {len(training_points)} training points and {len(test_points)} test points at overlap "
                  f"{stride_ms}ms and window {window_ms}ms")

            num_features = len(X_train[0])

            y_predict = mlp(X_train, y_train).predict(X_test)
            accuracies = get_metrics(y_test, y_predict)
            print_metrics(accuracies)
            res.append((window_ms, stride_ms, accuracies))
