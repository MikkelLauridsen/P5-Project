import datareader_csv
import metrics
from models.model_utility import get_scaled_test, get_scaled_training_validation
from run_models import selected_models, create_and_save_results


def __calculate_result_weight(result, f1_type='macro'):
    x = result.times['feature_time'] / 1000000
    y = result.times['model_time'] / 1000000
    z = result.metrics[f1_type].f1
    return x * -0.25 + y * -10 + z * 3.5


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


def sort_by_score(results):
    results.sort(key=__calculate_result_weight, reverse=True)


def get_best_for_models(results, models):
    best_results = []

    for model in models:
        model_results = metrics.filter_results(results, models=[model])
        sort_by_score(model_results)

        best_results.append(model_results[0])

    return best_results


if __name__ == '__main__':
    results = datareader_csv.load_all_results()
    results = metrics.filter_results(results, dos_types='modified')

    best_results = get_best_for_models(results, selected_models.keys())
    run_on_test(best_results)
