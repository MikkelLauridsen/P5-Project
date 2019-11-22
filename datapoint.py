"""
The DataPoint class describing an object containing the features for a single window of messages.

Functions:
is_header_matching: Compares the given header with the current attributes of the DataPoint class.
parse_csv_row: Parses a row in a csv file and returns a corresponding DataPoint object.
get_csv_row: Returns a list containing every attribute in the given DataPoint object.
"""
from recordclass import dataobject


class DataPoint(dataobject):
    """Class representing a single window of messages from the data through its features."""
    time_ms: float
    class_label: str
    mean_id_interval: float
    # variance_id_frequency: float              # Disabled for              modified
    # num_id_transitions: int                   # Disabled for original and modified
    # num_ids: int                              # Disabled for original and modified
    num_msgs: int                             # Disabled for original
    mean_id_intervals_variance: float
    mean_data_bit_count: float
    variance_data_bit_count: float
    # mean_variance_data_bit_count_id: float    # Disabled for original and modified
    # kurtosis_id_interval: float               # Disabled for              modified
    kurtosis_id_frequency: float
    skewness_id_frequency: float
    kurtosis_mean_id_intervals: float         # Disabled for original
    kurtosis_variance_data_bit_count_id: float
    skewness_id_interval_variances: float

    def __str__(self):
        string = ""

        for attr in datapoint_attributes:
            string += f"{attr}: {getattr(self, attr)} "

        return string


# List of the attributes of DataPoint
datapoint_attributes = list(DataPoint.__annotations__.keys())
datapoint_features = datapoint_attributes[2:]

# Descriptions of features. These are displayed on the plots.
datapoint_attribute_descriptions = {
    "mean_id_interval": "Mean id interval",
    "variance_id_frequency": "Variance id frequency",
    "num_id_transitions": "Number of id transitions",
    "num_ids": "Number of ids",
    "num_msgs": "Number of messages",
    "mean_id_intervals_variance":  "Mean id intervals variance",
    "mean_data_bit_count": "Mean data field bit-count",
    "variance_data_bit_count": "Data field bit-count variance",
    "mean_variance_data_bit_count_id": "Mean data field bit-count id variance",
    "mean_probability_bits": "Mean probability bits",
    "kurtosis_id_interval": "Kurtosis id interval",
    "kurtosis_id_frequency": "Kurtosis id frequencies",
    "skewness_id_frequency": "Skewness id frequencies",
    "kurtosis_mean_id_intervals": "Kurtosis mean id intervals",
    "kurtosis_variance_data_bit_count_id": "Kurtosis data field bit-count id variance",
    "skewness_id_interval_variances": "Skewness id interval variances"
}


def is_header_matching(header):
    """Compares the given header with the set of DataPoint attributes and returns the difference."""
    header_set = set(header)
    attr_set = set(datapoint_attributes)

    diff = header_set.symmetric_difference(attr_set)
    return len(diff) == 0, diff


def parse_csv_row(row):
    """Parses a single row from a csv file and returns the corresponding DataPoint object."""
    args = []
    for i in range(len(datapoint_attributes)):
        args.append(row[i])

    return DataPoint(*args)


def get_csv_row(datapoint):
    """Creates a list of the given DataPoints components and returns it to facilitate csv file creation."""
    row = []
    for attr in datapoint_attributes:
        row.append(getattr(datapoint, attr))

    return row
