from recordclass import dataobject


class IDPoint(dataobject):
    time_ms: float
    is_injected: bool
    mean_id_interval: float
    variance_id_frequency: float
    num_id_transitions: int
    num_ids: int
    num_msgs: int

    def __str__(self):
        return f"time_ms: {self.time_ms} injected: {self.is_injected} mean interval: {self.mean_id_interval} frequency variance: " \
            f"{self.variance_id_frequency} transitions: {self.num_id_transitions} ids: {self.num_ids} msgs: {self.num_msgs}"