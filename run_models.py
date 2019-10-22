from models.mlp import mlp
from id_based_datasets import load_or_create_datasets
from models.model_utility import scale_features, split_feature_label


def select_features(X, feature_index):
    X_sub = []

    for x in X:
        fs = []

        for i in range(len(x)):
            if i == feature_index:
                fs.append(x[i])

        X_sub.append(fs)

    return X_sub


def print_entry(entry):
    print(f"window_ms: {entry[0]} overlap_ms: {entry[1]}")

    for evaluator in entry[2]:
        print(f"\t {evaluator[0]}: {evaluator[1]}")

    print()


def get_accuracies(y_test, y_predict):
    tp = fp = tn = fn = 0

    if len(y_test) != len(y_predict):
        raise IndexError()

    for i in range(len(y_test)):
        if y_test[i] == "normal":
            if y_predict[i] == "normal":
                tn += 1
            else:
                fp += 1
        else:
            if y_predict[i] == "normal":
                fn += 1
            else:
                tp += 1

    precision = 1 if tp + fp == 0 else tp / (tp + fp)
    recall = tp / (tp + fn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    balanced_accuracy = (tpr + tnr) / 2
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    return [("precision", precision), ("recall", recall), ("TPR", tpr), ("TNR", tnr),
            ("balanced accuracy", balanced_accuracy), ("FPR", fpr), ("FNR", fnr)]


if __name__ == "__main__":

    res = []

    for window_ms in [10, 20, 30, 50, 80, 130]:
        for overlap_ms in [10, 20, 30, 50, 80, 130]:
            training_points, test_points = load_or_create_datasets(period_ms=window_ms, shuffle=True, overlap_ms=overlap_ms, dos_type='modified')
            X_train, y_train = split_feature_label(training_points)
            X_test, y_test = split_feature_label(test_points)
            X_train, X_test = scale_features(X_train, X_test)

            print(f"Generated {len(training_points)} training points and {len(test_points)} test points at overlap {overlap_ms}ms and window {window_ms}ms")

            num_features = len(X_train[0])

            y_predict = mlp(X_train, y_train).predict(X_test)
            accuracies = get_accuracies(y_test, y_predict)
            res.append((window_ms, overlap_ms, accuracies))
            print_entry((window_ms, overlap_ms, accuracies))

    for entry in res:
        print_entry(entry)
