import hugin.pyhugin87 as hugin
import os
import datapoint as dp
import models.model_utility as model_utility
from sklearn.metrics import classification_report
from datasets import load_or_create_datasets


class BayesianNetwork:
    __domain: hugin.DataSet
    __attr_names: []

    def __init__(self):
        self.__domain = hugin.Domain()

        self.__attr_names = list(dp.datapoint_attributes).copy()
        self.__attr_names.remove("time_ms")
        self.__attr_names.remove("is_injected")

    def fit(self, X_train, y_train):
        training_set = self.__feature_list_to_dataset(X_train, y_train, self.__attr_names)

        self.__construct_network_nodes(training_set)
        self.__learn_structure_and_tables(training_set)

    def predict(self, X_test, y_test):
        test_set = self.__feature_list_to_dataset(X_test, y_test, self.__attr_names)

        return self.__get_predictions(test_set)

    def save_network(self, name="bayesian_network.hkb"):
        # Save network to view in hugin
        self.__domain.save_as_kb(name)

    @staticmethod
    def __feature_list_to_dataset(features_list, injected_labels, names):
        dataset = hugin.DataSet()

        dataset.new_column("is_injected")  # Make sure is_injected is included in dataset

        # Create one column per feature
        for name in names:
            dataset.new_column(name)

        # Insert one row per set of features
        for row, features in enumerate(features_list):
            dataset.new_row()

            # Insert is_injected separately
            dataset.set_data_item(row, 0, injected_labels[row])

            # Insert features
            for col, value in enumerate(features):
                dataset.set_data_item(row, col + 1, str(value))

        return dataset

    def __get_predictions(self, dataset):
        self.__domain.set_number_of_cases(0)
        num_cases = dataset.get_number_of_rows()

        # Add dataset cases to domain
        self.__domain.add_cases(dataset, 0, num_cases)

        predictions = []
        for i in range(num_cases):
            self.__domain.reset_inference_engine()

            self.__domain.enter_case(i)  # Enter evidence from dataset into network

            injected_node = self.__domain.get_node_by_name("is_injected")
            injected_node.retract_findings()  # Remove known evidence about the is_injected node

            # Propagate evidence
            injected_node.get_junction_tree().propagate()

            normal_belief = injected_node.get_belief(0)
            fuzzy_belief = injected_node.get_belief(1)
            dos_belief = injected_node.get_belief(2)
            impersonation_belief = injected_node.get_belief(3)

            # Find state of is_injected node with largest value
            beliefs = [normal_belief, fuzzy_belief, dos_belief, impersonation_belief]
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

        # Insert node for is_injected. This is the only discrete node, and is therefore handled separately
        output_node = hugin.Node(self.__domain, kind=hugin.KIND.DISCRETE)
        output_node.set_name("is_injected")
        output_node.set_label("is_injected")
        output_node.set_number_of_states(4)
        output_node.set_state_label(0, "normal")
        output_node.set_state_label(1, "dos")
        output_node.set_state_label(2, "fuzzy")
        output_node.set_state_label(3, "impersonation")
        output_node.get_experience_table()

        # Add input nodes (one for each column in dataset). These are all continuous
        for name_index in range(dataset.get_number_of_columns()):
            name = dataset.get_column_name(name_index)

            # Ignore time_ms and is_injected attributes
            if name == "time_ms" or name == "is_injected":
                continue

            node = hugin.Node(self.__domain, kind=hugin.KIND.CONTINUOUS)
            node.set_name(name)
            node.set_label(name)
            node.get_experience_table()


    def __learn_structure_and_tables(self, dataset):
        # Add learning data from dataset to domain
        self.__domain.add_cases(dataset, 0, dataset.get_number_of_rows())

        # Learn structure and condition tables
        print("Learning structure...")
        self.__domain.learn_structure()
        self.__domain.compile()
        self.__domain.save_to_memory()
        print("Learning tables...")
        self.__domain.learn_tables()


if __name__ == "__main__":
    os.chdir("..")

    bn = BayesianNetwork()
    training_points, test_points, _ = load_or_create_datasets(impersonation_split=False, stride_ms=10, period_ms=10)
    X_train, y_train = model_utility.split_feature_label(training_points)
    bn.fit(X_train, y_train)

    X_test, y_test = model_utility.split_feature_label(test_points)
    predictions = bn.predict(X_test, y_test)

    print(classification_report(y_test, predictions))