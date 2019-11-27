dos_type = 'modified'
imp_type = 'imp_full'
imp_split = False
use_modified = True


def __set_modified():
    global dos_type, imp_type, imp_split, use_modified
    dos_type = 'modified'
    imp_type = 'imp_full'
    imp_split = False
    use_modified = True


def __set_original():
    global dos_type, imp_type, imp_split, use_modified
    dos_type = 'original'
    imp_type = 'imp_split'
    imp_split = True
    use_modified = False


__set_original()
#__set_modified()

selected_models = {
    'bn': {
        'significance_level': 0.5
    },

    'nbc': {},

    'mlp': {
        'activation': 'logistic',
        'alpha': 0.0001,
        'hidden_layer_sizes': (16, 3),
        'learning_rate': 'adaptive',
        'max_iter': 600,
        'solver': 'lbfgs'
    },

    'svm': {
        'C': 1,
        'kernel': 'linear'
    },

    'knn': {
        'metric': 'manhattan',
        'n_neighbors': 8,
        'weights': 'distance'
    },

    'lr': {
        'C': 3593.813663804626,
        'penalty': 'l2'
    },

    'dt': {
        'criterion': 'entropy',
        'max_depth': 13,
        'min_samples_split': 3
    },

    'rf': {
        'bootstrap': True,
        'criterion': 'gini',
        'max_depth': 11,
        'n_estimators': 110
    }
}
