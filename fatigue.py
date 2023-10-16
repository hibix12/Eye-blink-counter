import csv

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("tkagg")

with open('data3.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    values = [row for row in reader]

    data = [float(item[1]) for item in values]  # skip the header row, assuming you have one

    # PERCLOS counted for 1 minute (30fps = 1800 frames)

    sampling_rate = 30
    window_size = 30 * 60 * 4  # 2 minutes window

    thresholds = []
    initial_threshold = (np.median(data[:window_size]) + min(data)) / 2
    thresholds.extend([initial_threshold] * window_size)
    # Dla każdej kolejnej próbki
    for i in range(window_size, len(data)):
        threshold = (np.median(data[i - window_size:i]) + min(data[i - window_size:i])) / 2
        thresholds.append(threshold)

    frame_count = len(data)
    if frame_count > window_size:
        counter = []
        for i in range(frame_count):
            if data[i] > thresholds[i]:
                counter.append(1)
            else:
                counter.append(0)
        # windowing method, sum 1800 elements and then go to +1 one element
        part_sum = 0
        suma = []
        for i in range(window_size):
            part_sum += counter[i]
        suma.append(part_sum / window_size)
        for i in range(window_size, frame_count):
            part_sum -= counter[i - window_size]
            part_sum += counter[i]
            suma.append(part_sum / window_size)

        time_vector_seconds = np.linspace(window_size / sampling_rate, frame_count / sampling_rate,
                                          frame_count - window_size + 1)
        time_vector_minutes = time_vector_seconds / 60

        plt.plot(time_vector_minutes, suma)
        plt.xlabel("Czas [min]")
        plt.ylabel("PERCLOS [%]")
        plt.show()
    else:
        print("Not enough data")
