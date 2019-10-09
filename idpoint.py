from recordclass import dataobject


class IDPoint(dataobject):
    time_ms: float
    is_injected: str
    #mean_id_interval: float
    variance_id_frequency: float
    #num_id_transitions: int
    #num_ids: int
    #num_msgs: int
    #mean_id_intervals_variance: float
    #mean_data_bit_count: float
    # variance_data_bit_count: float
    #mean_variance_data_bit_count_id: float
    # mean_probability_bits: float
    #req_to_res_time_variance: float
    #kurtosis_id_interval: float
    kurtosis_id_frequency: float
    skewness_id_frequency: float
    #kurtosis_mean_id_intervals: float
    #kurtosis_variance_data_bit_count_id: float
    skewness_id_interval_variances: float
    kurtosis_mean_id_intervals: float
    kurtosis_req_to_res_time: float

    def __str__(self):
        str = ""

        for attr in idpoint_attributes:
            str += f"{attr}: {getattr(self, attr)}"

        return str


# List of the attributes of IDPoint
idpoint_attributes = IDPoint.__annotations__.keys()

# Descriptions of features. These are displayed on the plots.
idpoint_attribute_descriptions = {
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


def parse_csv_row(row):
    args = []
    for i in range(len(idpoint_attributes)):
        args.append(row[i])

    return IDPoint(*args)


def get_csv_row(idpoint):
    row = []
    for attr in idpoint_attributes:
        row.append(getattr(idpoint, attr))

    return row
