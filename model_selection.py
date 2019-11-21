import math

import datareader_csv
import metrics
from datapoint import datapoint_features
from models.model_utility import get_scaled_test, get_scaled_training_validation
from run_models import selected_models, create_and_save_results


def run_on_test(configurations):
    for configuration in configurations:
        X_test, y_test, feature_time_dict = get_scaled_test(
            configuration.period_ms,
            configuration.stride_ms,
            configuration.imp_split,
            configuration.dos_type)

        X_train, y_train, X_validation, y_validation, _ = get_scaled_training_validation(
            configuration.period_ms,
            configuration.stride_ms,
            configuration.imp_split,
            configuration.dos_type)

        X_train = list(X_validation) + list(X_train)
        y_train = list(y_validation) + list(y_train)

        create_and_save_results(
            configuration.model,
            selected_models[configuration.model],
            X_train, y_train,
            X_test, y_test,
            feature_time_dict,
            configuration.period_ms,
            configuration.stride_ms,
            configuration.imp_split,
            configuration.dos_type,
            configuration.subset,
            is_test=True)


def __calculate_result_weight(result, w_ft, w_mt, w_f1, f1_type):
    x = result.times['feature_time'] / 1000000
    y = result.times['model_time'] / 1000000
    z = result.metrics[f1_type].f1
    return x * w_ft + y * w_mt + z * w_f1


def __sort_by_score(results, w_ft, w_mt, w_f1, f1_type):
    sort_func = lambda x: __calculate_result_weight(x, w_ft, w_mt, w_f1, f1_type)
    results.sort(key=sort_func, reverse=True)


def get_best_for_models(results, models, w_ft=-0.25, w_mt=-10, w_f1=3.5, f1_type='macro'):
    """
    Returns a list of the best results. One for each model
    :param results: A list of results to search through
    :param models: A list of model labels to find best models for
    :param w_ft: The weight of feature time
    :param w_mt: The weight of model time
    :param w_f1: The weight of f1 score
    :param f1_type: f1 type
    :return:
    """
    best_results = []

    for model in models:
        model_results = metrics.filter_results(results, models=[model])
        __sort_by_score(model_results, w_ft, w_mt, w_f1, f1_type)

        best_results.append(model_results[0])

    return best_results


def get_feature_statistics(results):
    to_be_deleted = []

    for result in results:
        if len(result.subset) != 6:
            to_be_deleted.append(result)

    for result in to_be_deleted:
        pass#del results[results.index(result)]

    length = len(results)
    feature_labels = datapoint_features
    statistics = {}

    for label in feature_labels:
        result_with = metrics.filter_results(results, features=[label])
        result_without = metrics.filter_results(results, without_features=[label])

        with_length = len(result_with)
        without_length = len(result_without)
        prevalence = with_length / length

        if prevalence != 0:
            avg_f1_dos = math.fsum([result.metrics['dos'].f1 for result in result_with]) / with_length
            avg_f1_fuzzy = math.fsum([result.metrics['fuzzy'].f1 for result in result_with]) / with_length
            avg_f1_imp = math.fsum([result.metrics['impersonation'].f1 for result in result_with]) / with_length
        else:
            avg_f1_dos = 0
            avg_f1_fuzzy = 0
            avg_f1_imp = 0

        avg_f1_without_dos = math.fsum([result.metrics['dos'].f1 for result in result_without]) / without_length
        avg_f1_without_fuzzy = math.fsum([result.metrics['fuzzy'].f1 for result in result_without]) / without_length
        avg_f1_without_imp = math.fsum([result.metrics['impersonation'].f1 for result in result_without]) / without_length
        avg_f1_diff_dos = avg_f1_without_dos - avg_f1_dos
        avg_f1_diff_fuzzy = avg_f1_without_fuzzy - avg_f1_fuzzy
        avg_f1_diff_imp = avg_f1_without_imp - avg_f1_imp

        statistics[label] = [prevalence, avg_f1_diff_dos, avg_f1_diff_fuzzy, avg_f1_diff_imp]

    return statistics


if __name__ == '__main__':
    results = datareader_csv.load_all_results()
    results = metrics.filter_results(results, dos_types=['modified'], imp_splits=[False], periods=[100])

    statistics = get_feature_statistics(results)

    for statistic in statistics.keys():
        print("{}: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(statistic, statistics[statistic][0], statistics[statistic][1], statistics[statistic][2], statistics[statistic][3]))

    best_results = get_best_for_models(results, selected_models.keys())
    run_on_test(best_results)
