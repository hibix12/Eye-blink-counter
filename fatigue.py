import csv

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("tkagg")

with open('data3.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    values = [row for row in reader]

    data = [float(item[1]) for item in values]
    sampling_rate = 30
    window_size = sampling_rate * 30

    thresholds = []
    initial_threshold = (np.median(data[:window_size]) + min(data)) / 2
    thresholds.extend([initial_threshold] * window_size)
    # Dla każdej kolejnej próbki
    for i in range(window_size, len(data)):
        threshold = (np.median(data[i - window_size:i]) + min(data[i - window_size:i])) / 2
        thresholds.append(threshold)

    window_perclos = sampling_rate * 60 * 4

    frame_count = len(data)
    if frame_count > window_perclos:
        counter = []
        for i in range(frame_count):
            if data[i] > thresholds[i]:
                counter.append(0)
            else:
                counter.append(1)
        part_sum = 0
        suma = []
        for i in range(window_perclos):
            part_sum += counter[i]
        suma.append(part_sum / window_perclos)
        for i in range(window_perclos, frame_count):
            part_sum -= counter[i - window_perclos]
            part_sum += counter[i]
            suma.append(part_sum / window_perclos)

        time_vector_seconds = np.linspace(window_perclos / sampling_rate, frame_count / sampling_rate,
                                          frame_count - window_perclos + 1)
        time_vector_minutes = time_vector_seconds / 60
        perclos = [s * 100 for s in suma]
        plt.plot(time_vector_minutes, perclos)
        plt.xlabel("Czas [min]")
        plt.ylabel("PERCLOS [%]")
        plt.show()
    else:
        print("Not enough data")
