from recordclass import dataobject


class IDPoint(dataobject):
    time_ms: float
    is_injected: str
    mean_id_interval: float
    variance_id_frequency: float
    num_id_transitions: int
    num_ids: int
    num_msgs: int
    mean_id_intervals_variance: float
    mean_data_bit_count: float
    variance_data_bit_count: float
    mean_variance_data_bit_count_id: float
    mean_probability_bits: float
    req_to_res_time_variance: float
    kurtosis_id_frequency: float

    def __str__(self):
        str = ""

        for attr in idpoint_attributes:
            str += f"{attr}: {getattr(self, attr)}"

        return str

# Must be same length and order as attributes in IDPoint
idpoint_attributes = ["time_ms", "is_injected", "mean_id_interval", "variance_id_frequency",
                  "num_id_transitions", "num_ids", "num_msgs", "mean_id_intervals_variance",
                  "mean_data_bit_count", "variance_data_bit_count",
                  "mean_variance_data_bit_count_id", "mean_probability_bits",
                  "req_to_res_time_variance", "kurtosis_id_frequency"]


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
