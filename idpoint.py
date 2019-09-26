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

    def __str__(self):
        return f"time_ms: {self.time_ms} injected: {self.is_injected} mean interval: {self.mean_id_interval} frequency variance: " \
            f"{self.variance_id_frequency} transitions: {self.num_id_transitions} ids: {self.num_ids} msgs: {self.num_msgs}"


def parse_csv_row(row):
    time_ms = float(row[0])
    is_injected = True if row[1] == "True" else False
    mean_id_interval = float(row[2])
    variance_id_frequency = float(row[3])
    num_id_transitions = int(row[4])
    num_ids = int(row[5])
    num_msgs = int(row[6])
    mean_id_intervals_variance = float(row[7])

    return IDPoint(time_ms, is_injected, mean_id_interval, variance_id_frequency, num_id_transitions, num_ids, num_msgs, mean_id_intervals_variance)


def get_csv_row(idpoint):
    return [str(idpoint.time_ms),
            str(idpoint.is_injected),
            str(idpoint.mean_id_interval),
            str(idpoint.variance_id_frequency),
            str(idpoint.num_id_transitions),
            str(idpoint.num_ids),
            str(idpoint.num_msgs),
            str(idpoint.mean_id_intervals_variance)]