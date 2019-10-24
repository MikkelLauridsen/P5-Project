from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import datapoint


# Splitting the training data into feature and label lists.
def split_feature_label(datapoints):
    # The instance list that will contain all the features for each instance.
    X = []
    # The label list that will contain the injected status for each idpoint.
    y = []

    # Going through each instance and extracting the features and labels.
    for point in datapoints:
        features = []
        for attr in datapoint.datapoint_attributes:
            if attr == "time_ms":
                pass
            elif attr == "is_injected":
                y.append(point.is_injected)
            else:
                features.append(getattr(point, attr))

        X.append(features)

    return X, y


# Fitting a transformation on the training features and scaling both the training features and test features to it.
def scale_features(X_training, X_test):
    # fit the scaler
    scaler = StandardScaler()
    scaler.fit(X_training)

    # Transforming the training and test features with the fitted scaler.
    X_training = scaler.transform(X_training)
    X_test = scaler.transform(X_test)

    return X_training, X_test


# Finds the best combination of hyperparameters and prints the results.
def find_best_hyperparameters(estimator, parameter_grid, X_train, y_train, X_test, y_test):
    # Creating the grid search that uses the parameter grid to find the best hyperparameters.
    grid_s = GridSearchCV(estimator, parameter_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)
    grid_s.fit(X_train, y_train)

    print(f"parameters found: {grid_s.best_params_}")

    y_predict = grid_s.predict(X_test)

    return y_predict


def best_hyper_parameters_for_all_model(estimator, parameter_grid, X_train, y_train):
    # Creating the grid search that uses the parameter grid to find the best hyperparameters.
    grid_s = GridSearchCV(estimator, parameter_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)
    grid_s.fit(X_train, y_train)

    print(f"parameters found: {grid_s.best_params_}")
    return grid_s


def print_metrics(metric_dic):
    print("printing classification metrics:")

    labels = ['Precision', 'Recall', 'TPR', 'TNR', 'FPR', 'FNR', 'Balanced accuracy', 'F1-score']
    print("{:15}".format(" ") + "".join(["{:>18}".format(f"{label} ") for label in labels]))

    for key in metric_dic.keys():
        line = "{:15}".format(f"{key}: ")

        for i in range(len(labels)):
            line += "{:17.4f}".format(metric_dic[key][i]) + " "

        print(line)


def get_metrics(y_test, y_predict):
    tp = fp = tn = fn = 0
    class_counters = {'normal': [0, 0, 0, 0], 'dos': [0, 0, 0, 0], 'fuzzy': [0, 0, 0, 0], 'impersonation': [0, 0, 0, 0]}

    if len(y_test) != len(y_predict):
        raise IndexError()

    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            __increment_equal(y_test[i], class_counters)

            if y_test[i] == 'normal':
                tn += 1
            else:
                tp += 1
        else:
            __increment_not_equal(y_test[i], y_predict[i], class_counters)

            if y_test[i] == 'normal':
                fn += 1
            else:
                fp += 1

    metrics = {'total': __get_metrics_tuple(tp, fp, tn, fn)}

    for key in class_counters.keys():
        counts = class_counters[key]
        metrics[key] = __get_metrics_tuple(counts[0], counts[1], counts[2], counts[3])

    return metrics


def __increment_equal(label, class_counters):
    class_counters[label][0] += 1

    for key in class_counters.keys():
        if key != label:
            class_counters[key][2] += 1


def __increment_not_equal(label, prediction, class_counters):
    class_counters[label][3] += 1
    class_counters[prediction][1] += 1

    for key in class_counters.keys():
        if key != label and key != prediction:
            class_counters[key][2] += 1


def __get_metrics_tuple(tp, fp, tn, fn):
    precision = 1 if tp + fp == 0 else tp / (tp + fp)
    recall = tp / (tp + fn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    balanced_accuracy = (tpr + tnr) / 2
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, tpr, tnr, fpr, fnr, balanced_accuracy, f1