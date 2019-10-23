import hugin.pyhugin87 as hugin
import os
import datapoint as dp
import models.model_utility as model_utility
from sklearn.metrics import classification_report
from datasets import load_or_create_datasets


# Creates a hugin dataset from a list of sets of features and their labels
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


# Creates a Hugin dataset from a list of DataPoints
def __datapoints_to_dataset(datapoints):
    dataset = hugin.DataSet()

    dataset.new_column("is_injected")

    # Create column for each attribute
    for attr in dp.datapoint_attributes:
        dataset.new_column(attr)

    # Create row for each DataPoint
    for row, point in enumerate(datapoints):
        dataset.new_row()

        # Insert attributes from DataPoint into row
        for col, attr in enumerate(dp.datapoint_attributes):
            value = getattr(point, attr)
            dataset.set_data_item(row, col, str(value))

    return dataset


# Prints a Hugin dataset. Used for debugging
def __print_dataset(dataset, limit=0):
    num_cols = dataset.get_number_of_columns()
    num_rows = dataset.get_number_of_rows()

    if limit == 0:
        limit = num_rows

    # Print columns
    for c in range(num_cols):
        col_name = dataset.get_column_name(c)
        print(f"{col_name}, ", end="")
    print()

    # Print rows
    for r in range(num_rows):
        if r == limit:
            break

        for c in range(num_cols):
            value = dataset.get_data_item(r, c)
            if c == 0:  # c == 0 is the case for is_injected
                print(value + ", ", end="")
            else:
                print(f"%0.2f, " % float(value), end="")
        print()


# Constructs the nodes of a network from a given dataset
def __construct_network_nodes(domain, dataset):

    # Insert node for is_injected. This is the only discrete node, and is therefore handled separately
    output_node = hugin.Node(domain, kind=hugin.KIND.DISCRETE)
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

        node = hugin.Node(domain, kind=hugin.KIND.CONTINUOUS)
        node.set_name(name)
        node.set_label(name)
        node.get_experience_table()


def __learn_structure_and_tables(domain, dataset):
    # Add learning data from dataset to domain
    domain.add_cases(dataset, 0, dataset.get_number_of_rows())

    # Learn structure and condition tables
    print("Learning structure...")
    domain.learn_structure()
    domain.compile()
    domain.save_to_memory()
    print("Learning tables...")
    domain.learn_tables()


def __get_predictions(domain, dataset):
    domain.set_number_of_cases(0)
    num_cases = dataset.get_number_of_rows()

    # Add dataset cases to domain
    domain.add_cases(dataset, 0, num_cases)

    predictions = []
    for i in range(num_cases):
        domain.reset_inference_engine()

        domain.enter_case(i)  # Enter evidence from dataset into network

        injected_node = domain.get_node_by_name("is_injected")
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


# Trains a bayesian network and tests a dataset on it
def train_and_predict(training_points, test_points):
    X_train, y_train = model_utility.split_feature_label(training_points)
    X_test, y_test = model_utility.split_feature_label(test_points)

    # Create hugin datasets. names corresponds to the names of the features in training_points
    names = list(dp.datapoint_attributes).copy()
    names.remove("time_ms")
    names.remove("is_injected")
    training_set = __feature_list_to_dataset(X_train, y_train, names)
    test_set = __feature_list_to_dataset(X_test, y_test, names)

    # Create hugin domain and train on training_set and predict on training_set
    domain = hugin.Domain()
    __construct_network_nodes(domain, training_set)
    __learn_structure_and_tables(domain, training_set)

    # Save network to view in hugin
    # domain.save_as_kb("bayesian_network.hkb")

    return __get_predictions(domain, test_set)


if __name__ == "__main__":
    os.chdir("..")

    training_points, test_points = load_or_create_datasets()
    X_test, y_test = model_utility.split_feature_label(test_points)

    predictions = train_and_predict(training_points, test_points)

    print(classification_report(y_test, predictions))