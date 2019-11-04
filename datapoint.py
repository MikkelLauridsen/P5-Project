from recordclass import dataobject


class DataPoint(dataobject):
    time_ms: float
    is_injected: str
    mean_id_interval: float
    variance_id_frequency: float
    # num_id_transitions: int
    # num_ids: int
    # num_msgs: int
    mean_id_intervals_variance: float
    mean_data_bit_count: float
    variance_data_bit_count: float
    # mean_variance_data_bit_count_id: float
    kurtosis_id_interval: float
    kurtosis_id_frequency: float
    skewness_id_frequency: float
    # kurtosis_mean_id_intervals: float
    kurtosis_variance_data_bit_count_id: float
    skewness_id_interval_variances: float

    def __str__(self):
        str = ""

        for attr in datapoint_attributes:
            str += f"{attr}: {getattr(self, attr)} "

        return str


# List of the attributes of DataPoint
datapoint_attributes = list(DataPoint.__annotations__.keys())

# Descriptions of features. These are displayed on the plots.
datapoint_attribute_descriptions = {
    "mean_id_interval": "Mean id interval",
    "variance_id_frequency": "Variance id frequency",
    "num_id_transitions": "# id transitions",
    "num_ids": "# ids",
    "num_msgs": "# messages",
    "mean_data_bit_count": "Mean data bit-counts",
    "variance_data_bit_count": "Variance data bit-counts",
    "mean_variance_data_bit_count_id": "Mean variance data bit-count ids",
    "mean_probability_bits": "Mean probability bits",
    "kurtosis_variance_data_bit_count_id": "Kurtosis variance data bit-count",
    "kurtosis_id_frequency": "Kurtosis id frequencies",
    "skewness_id_frequency": "Skewness id frequencies",
    "skewness_id_interval_variances": "Skewness id interval variances"
}


def is_header_matching(header):
    header_set = set(header)
    attr_set = set(datapoint_attributes)

    diff = header_set.symmetric_difference(attr_set)
    return len(diff) == 0, diff


def parse_csv_row(row):
    args = []
    for i in range(len(datapoint_attributes)):
        args.append(row[i])

    return DataPoint(*args)


def get_csv_row(datapoint):
    row = []
    for attr in datapoint_attributes:
        row.append(getattr(datapoint, attr))

    return row
