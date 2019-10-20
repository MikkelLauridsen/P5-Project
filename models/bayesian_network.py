import hugin.pyhugin87 as hugin
import os
import idpoint as idp
import datareader_csv
import models.model_utility as model_utility
from sklearn.metrics import classification_report


def __feature_list_to_dataset(features_list, injected_labels, names):
    dataset = hugin.DataSet()

    dataset.new_column("is_injected")

    for name in names:
        dataset.new_column(name)

    for row, features in enumerate(features_list):
        dataset.new_row()

        dataset.set_data_item(row, 0, injected_labels[row])

        for col, value in enumerate(features):
            dataset.set_data_item(row, col + 1, str(value))

    return dataset


def __idpoints_to_dataset(idpoints):
    dataset = hugin.DataSet()

    dataset.new_column("is_injected")

    # Create column for each attribute
    for attr in idp.idpoint_attributes:
        dataset.new_column(attr)

    # Create row for each idpoint
    for row, idpoint in enumerate(idpoints):
        dataset.new_row()

        # Insert attributes from idpoint into row
        for col, attr in enumerate(idp.idpoint_attributes):
            value = getattr(idpoint, attr)
            dataset.set_data_item(row, col, str(value))

    return dataset


def __print_dataset(dataset, limit=0):
    num_cols = dataset.get_number_of_columns()
    num_rows = dataset.get_number_of_rows()

    if limit == 0:
        limit = num_rows

    for c in range(num_cols):
        col_name = dataset.get_column_name(c)
        print(f"{col_name}, ", end="")
    print()

    for r in range(num_rows):
        if r == limit:
            break

        for c in range(num_cols):
            value = dataset.get_data_item(r, c)
            if c == 0:
                print(value + ", ", end="")
            else:
                print(f"%0.2f, " % float(value), end="")
        print()


def __construct_network_nodes(domain, dataset):
    # Add input nodes (one for each column in dataset)
    for name_index in range(dataset.get_number_of_columns()):
        name = dataset.get_column_name(name_index)

        if name == "time_ms" or name == "is_injected":
            continue

        node = hugin.Node(domain, kind=hugin.KIND.CONTINUOUS)
        node.set_name(name)
        node.set_label(name)
        node.get_experience_table()

    output_node = hugin.Node(domain, kind=hugin.KIND.DISCRETE)
    output_node.set_name("is_injected")
    output_node.set_label("is_injected")
    output_node.set_number_of_states(4)
    output_node.set_state_label(0, "normal")
    output_node.set_state_label(1, "dos")
    output_node.set_state_label(2, "fuzzy")
    output_node.set_state_label(3, "impersonation")
    output_node.get_experience_table()


def __learn_structure_and_tables(domain, dataset):
    # Add learning data from dataset to domain
    domain.add_cases(training_set, 0, training_set.get_number_of_rows())

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
    domain.add_cases(dataset, 0, num_cases)

    predictions = []
    for i in range(num_cases):
        domain.reset_inference_engine()

        for j, attr in enumerate(idp.idpoint_attributes):
            if attr == "time_ms" or attr == "is_injected":
                continue

            value = float(dataset.get_data_item(i, j - 1))
            domain.get_node_by_name(attr).enter_value(value)

        #domain.enter_case(i)
        injected_node = domain.get_node_by_name("is_injected")
        #injected_node.unset_case(i)

        injected_node.get_junction_tree().propagate()

        normal_belief = injected_node.get_belief(0)
        fuzzy_belief = injected_node.get_belief(1)
        dos_belief = injected_node.get_belief(2)
        impersonation_belief = injected_node.get_belief(3)

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


if __name__ == "__main__":
    os.chdir("..")

    training_points = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_training_37582_100ms.csv", 0)
    test_points = datareader_csv.load_idpoints("data/idpoint_dataset/mixed_test_9396_100ms.csv", 0)

    X_train, y_train = model_utility.split_feature_label(training_points)
    X_test, y_test = model_utility.split_feature_label(test_points)

    names = list(idp.idpoint_attributes).copy()
    names.remove("time_ms")
    names.remove("is_injected")
    training_set = __feature_list_to_dataset(X_train, y_train, names)
    test_set = __feature_list_to_dataset(X_test, y_test, names)

    domain = hugin.Domain()

    __construct_network_nodes(domain, training_set)
    __learn_structure_and_tables(domain, training_set)

    #domain.save_as_kb("network.hkb")

    predictions = __get_predictions(domain, test_set)
    print(classification_report(y_test, predictions))