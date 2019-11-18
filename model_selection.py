import datareader_csv
import metrics
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


if __name__ == '__main__':
    results = datareader_csv.load_all_results()
    results = metrics.filter_results(results, dos_types='modified')

    best_results = get_best_for_models(results, selected_models.keys())
    run_on_test(best_results)
