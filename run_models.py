import os
from datapoint import datapoint_attributes
from datasets import load_or_create_datasets
from models.model_utility import scale_features, split_feature_label, get_metrics, save_metrics, save_time, get_classifier, get_metrics_path, find_best_hyperparameters, load_metrics


def generate_results(windows=[100], strides=[100], imp_splits=[True],
                     dos_types=['modified'], models={'mlp': {}}, feature_steps=4):

    inner_loop_size = get_stepwise_size(feature_steps) * len(models)
    job_count = len(windows) * len(strides) * len(imp_splits) * len(dos_types) * inner_loop_size
    current_job = 0

    for period_ms in windows:
        for stride_ms in strides:
            for imp_split in imp_splits:
                for dos_type in dos_types:
                    print(f"starting jobs {current_job} through {current_job + inner_loop_size} of "
                          f"{job_count} -- {current_job/job_count}%")

                    X_train, y_train, X_test, y_test, feature_time_dict = get_dataset(period_ms, stride_ms, imp_split, dos_type)
                    save_stepwise_addition(models, X_train, y_train, X_test, y_test, feature_steps, feature_time_dict, period_ms, stride_ms, imp_split, dos_type)


def get_stepwise_size(max_features):
    n = len(list(datapoint_attributes)[2:])
    size = 0

    for i in range(max_features):
        size += n
        n -= 1

    return size


def get_dataset(period_ms, stride_ms, imp_split, dos_type):
    training_data, test_data, feature_time_dict = load_or_create_datasets(period_ms, True, stride_ms,
                                                                          imp_split, dos_type, verbose=True)

    X_train, y_train = split_feature_label(training_data)
    X_test, y_test = split_feature_label(test_data)
    X_train, X_test = scale_features(X_train, X_test)

    return X_train, y_train, X_test, y_test, feature_time_dict


def save_stepwise_addition(models, X_train, y_train, X_test, y_test, max_features, feature_time_dict, period_ms, stride_ms, imp_split, dos_type):
    labels = (list(datapoint_attributes)[2:]).copy()
    working_set = labels.copy()

    for i in range(0, max_features):
        best_score = 0
        best_label = ""

        for label in labels:
            for model in models.keys():
                current_subset = working_set.copy()
                del current_subset[current_subset.index(label)]

                path, _ = get_metrics_path(period_ms, stride_ms, imp_split, dos_type, model, models[model], current_subset)

                if os.path.exists(path):
                    metrics = load_metrics(period_ms, stride_ms, imp_split, dos_type, model, models[model], working_set + [label])
                else:
                    X_train_mod = create_feature_subset(X_train, current_subset)
                    X_test_mod = create_feature_subset(X_test, current_subset)
                    y_predict, time_model = find_best_hyperparameters(get_classifier(model), models[model], X_train_mod, y_train, X_test_mod)
                    metrics = get_metrics(y_test, y_predict)

                    save_metrics(metrics, period_ms, stride_ms, imp_split, dos_type, model, models[model], current_subset)
                    time_feature = 0.0

                    for feature in feature_time_dict.keys():
                        if feature in current_subset:
                            time_feature += feature_time_dict[feature]

                    save_time(time_model, time_feature, period_ms, stride_ms, imp_split, dos_type, model, models[model], current_subset)

                if metrics['total'][6] > best_score:
                    best_label = label

        del working_set[working_set.index(best_label)]
        del labels[labels.index(best_label)]


def create_feature_subset(X, subset):
    indices = [list(datapoint_attributes)[2:].index(f) for f in subset]
    length = len(list(datapoint_attributes)[2:])

    X_mod = []

    for sample in X:
        sample_mod = []

        for i in range(0, length):
            if i in indices:
                sample_mod.append(sample[i])

        X_mod.append(sample_mod)

    return X_mod


if __name__ == "__main__":
    generate_results()
