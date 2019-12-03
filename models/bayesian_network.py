import os

import hugin.pyhugin87 as hugin
from sklearn.base import BaseEstimator
import datapoint


class BayesianNetwork(BaseEstimator):
    __domain: hugin.DataSet
    __attr_names: []
    __output_node: hugin.Node
    __input_nodes: []
    __significance_level: float
    classes_: [] = ["normal", "dos", "fuzzy", "impersonation"]

    def __init__(self, subset, significance_level=0.05):
        self.__domain = hugin.Domain()
        self.__attr_names = subset
        self.__input_nodes = []
        self.__significance_level = significance_level

    def fit(self, X_train, y_train):
        training_set = self.__feature_list_to_dataset(X_train, y_train, self.__attr_names)

        self.__construct_network_nodes(training_set)
        self.__learn_structure_and_tables(training_set)

    def predict(self, features_list):
        labels = ["normal" for _ in features_list]  # Create fictional labels. This is ignored in prediction
        test_set = self.__feature_list_to_dataset(features_list, labels, self.__attr_names)

        return self.__get_predictions(test_set)

    def predict_proba(self, features_list):
        labels = ["normal" for _ in features_list]  # Create fictional labels. This is ignored in prediction
        test_set = self.__feature_list_to_dataset(features_list, labels, self.__attr_names)

        return self.__get_probabilities(test_set)

    def set_params(self, **params):
        self.__significance_level = params.get('significance_level', self.__significance_level)
        self.__attr_names = params.get('subset', self.__attr_names)
        return self

    def get_params(self, deep=True):
        return {
            'significance_level': self.__significance_level,
            'subset': self.__attr_names
        }

    def save_network(self, name="bayesian_network.hkb"):
        # Save network to view in hugin
        self.__domain.save_as_kb(name)

    @staticmethod
    def __feature_list_to_dataset(features_list, injected_labels, names):
        dataset = hugin.DataSet()

        dataset.new_column("class_label")  # Make sure class_label is included in dataset

        # Create one column per feature
        for name in names:
            dataset.new_column(name)

        # Insert one row per set of features
        for row, features in enumerate(features_list):
            dataset.new_row()

            # Insert class_label separately
            dataset.set_data_item(row, 0, injected_labels[row])

            # Insert features
            for col, value in enumerate(features):
                dataset.set_data_item(row, col + 1, str(value))

        return dataset

    def __get_probabilities(self, dataset):
        self.__domain.set_number_of_cases(0)
        num_cases = dataset.get_number_of_rows()

        # Add dataset cases to domain
        self.__domain.add_cases(dataset, 0, num_cases)

        probabilites = []
        for i in range(num_cases):
            self.__domain.reset_inference_engine()

            self.__domain.enter_case(i)  # Enter evidence from dataset into network

            injected_node = self.__domain.get_node_by_name("class_label")
            injected_node.retract_findings()  # Remove known evidence about the class_label node

            # Propagate evidence
            injected_node.get_junction_tree().propagate()

            normal_belief = injected_node.get_belief(0)
            fuzzy_belief = injected_node.get_belief(1)
            dos_belief = injected_node.get_belief(2)
            impersonation_belief = injected_node.get_belief(3)

            # Find state of class_label node with largest value
            probabilites.append([normal_belief, fuzzy_belief, dos_belief, impersonation_belief])

        return probabilites

    def __get_predictions(self, dataset):
        probabilities = self.__get_probabilities(dataset)

        predictions = []
        for beliefs in probabilities:
            largest_index = beliefs.index(max(beliefs))

            if largest_index == 0:
                predictions.append("normal")
            elif largest_index == 1:
                predictions.append("dos")
            elif largest_index == 2:
                predictions.append("fuzzy")
            else:
                predictions.append("impersonation")

        return predictions

    # Constructs the nodes of a network from a given dataset
    def __construct_network_nodes(self, dataset):
        self.__domain.set_significance_level(self.__significance_level)

        # Insert node for class_label. This is the only discrete node, and is therefore handled separately
        output_node = hugin.Node(self.__domain, kind=hugin.KIND.DISCRETE)
        output_node.set_name("class_label")
        output_node.set_label("class_label")
        output_node.set_number_of_states(4)
        output_node.set_state_label(0, "normal")
        output_node.set_state_label(1, "dos")
        output_node.set_state_label(2, "fuzzy")
        output_node.set_state_label(3, "impersonation")
        output_node.get_experience_table()
        self.__output_node = output_node

        # Add input nodes (one for each column in dataset). These are all continuous
        for name_index in range(dataset.get_number_of_columns()):
            name = dataset.get_column_name(name_index)

            # Ignore time_ms and class_label attributes
            if name == "time_ms" or name == "class_label":
                continue

            node = hugin.Node(self.__domain, kind=hugin.KIND.CONTINUOUS)
            node.set_name(name)
            node.set_label(name)
            node.get_experience_table()
            self.__input_nodes.append(node)

    def __learn_structure_and_tables(self, dataset):
        # Add learning data from dataset to domain
        self.__domain.add_cases(dataset, 0, dataset.get_number_of_rows())

        # Learn structure and condition tables
        self.__domain.learn_structure()
        #self.__domain.learn_tree_structure(self.__input_nodes[0], self.__output_node)
        self.__domain.compile()
        self.__domain.save_to_memory()
        self.__domain.learn_tables()


def bn(params={}):
    return BayesianNetwork(datapoint.datapoint_features).set_params(**params)


if __name__ == '__main__':
    from models.model_utility import get_standard_feature_split
    from sklearn.model_selection import GridSearchCV

    os.chdir("..")

    parameter_space = {
        'subset': [datapoint.datapoint_features],
        'significance_level': [0.1, 0.05, 0.01, 0.001]
    }

    X_train, y_train = get_standard_feature_split()

    # Find the hyperparameter values
    grid_s = GridSearchCV(bn(), parameter_space, cv=2, n_jobs=1, scoring="f1_macro", verbose=10)
    grid_s.fit(X_train, y_train)

    grid_s.best_estimator_.save_network("network.hkb")

    print(grid_s.best_params_)
    print(grid_s.best_score_)