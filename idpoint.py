from recordclass import dataobject


class IDPoint(dataobject):
    time_ms: float
    is_injected: bool
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

    def __str__(self):
        return f"time_ms: {self.time_ms} injected: {self.is_injected} mean interval: {self.mean_id_interval} frequency variance: " \
            f"{self.variance_id_frequency} transitions: {self.num_id_transitions} ids: {self.num_ids} msgs: {self.num_msgs}"


csv_header_row = ["time_ms", "is_injected", "mean_id_interval", "variance_id_frequency",
                  "num_id_transitions", "num_ids", "num_msgs", "mean_id_intervals_variance",
                  "mean_data_bit_count", "variance_data_bit_count",
                  "mean_variance_data_bit_count_id", "mean_probability_bits"]


def parse_csv_row(row):
    time_ms = row[0]
    is_injected = row[1]
    mean_id_interval = row[2]
    variance_id_frequency = row[3]
    num_id_transitions = row[4]
    num_ids = row[5]
    num_msgs = row[6]
    mean_id_intervals_variance = row[7]
    mean_data_bit_count = row[8]
    variance_data_bit_count = row[9]
    mean_variance_data_bit_count_id = row[10]
    mean_probability_bits = row[11]

    return IDPoint(time_ms, is_injected, mean_id_interval, variance_id_frequency,
                   num_id_transitions, num_ids, num_msgs, mean_id_intervals_variance,
                   mean_data_bit_count, variance_data_bit_count,
                   mean_variance_data_bit_count_id, mean_probability_bits)


def get_csv_row(idpoint):
    return [str(idpoint.time_ms),
            str(idpoint.is_injected),
            str(idpoint.mean_id_interval),
            str(idpoint.variance_id_frequency),
            str(idpoint.num_id_transitions),
            str(idpoint.num_ids),
            str(idpoint.num_msgs),
            str(idpoint.mean_id_intervals_variance),
            str(idpoint.mean_data_bit_count),
            str(idpoint.variance_data_bit_count),
            str(idpoint.mean_variance_data_bit_count_id),
            str(idpoint.mean_probability_bits)]