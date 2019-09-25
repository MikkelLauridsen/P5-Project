import datareader_csv
import matplotlib.pyplot as plt


idpoints = datareader_csv.load_idpoints("idpoint_dataset/mixed_training_32888_100ms.csv", 0)
print(idpoints[0])


time_ms = [idpoint.time_ms for idpoint in idpoints]
is_injected = [idpoint.is_injected for idpoint in idpoints]
mean_id_interval = [idpoint.mean_id_interval for idpoint in idpoints]
variance_id_frequency = [idpoint.variance_id_frequency for idpoint in idpoints]
num_id_transitions = [idpoint.num_id_transitions for idpoint in idpoints]
num_ids = [idpoint.num_ids for idpoint in idpoints]
num_msgs = [idpoint.num_msgs for idpoint in idpoints]


plt.figure(figsize=(20, 10))
plt.scatter(time_ms, mean_id_interval)
plt.xlabel("Time")
plt.ylabel("Median id interval")
plt.show()

plt.figure(figsize=(20, 10))
plt.scatter(time_ms, variance_id_frequency)
plt.xlabel("Time")
plt.ylabel("Variance id frequency")
plt.show()

plt.figure(figsize=(20, 10))
plt.scatter(time_ms, num_id_transitions)
plt.xlabel("Time")
plt.ylabel("# id transitions")
plt.show()

plt.figure(figsize=(20, 10))
plt.scatter(time_ms, num_ids)
plt.xlabel("Time")
plt.ylabel("# ids")
plt.show()

plt.figure(figsize=(20, 10))
plt.scatter(time_ms, num_msgs)
plt.xlabel("Time")
plt.ylabel("# messages")
plt.show()

plt.figure(figsize=(20, 10))
plt.stackplot(time_ms, is_injected)
plt.xlabel("Time")
plt.ylabel("Is injected")
plt.show()
