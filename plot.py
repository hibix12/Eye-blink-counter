import csv

import scipy
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator

matplotlib.use("tkagg")


with open('data1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    values = [row for row in reader]
    data = [float(item[1]) for item in values if 0.1 < float(item[1]) < 0.4]
    min_val = min(data)
    max_val = max(data)
    normalization = [(x - min_val) / (max_val - min_val) for x in data]
    data = normalization

    window_size = 30*30
    thresholds = []
    initial_threshold = (np.median(data[:window_size]) + np.percentile(data, 1)) / 2
    thresholds.extend([initial_threshold] * window_size)
    # Dla każdej kolejnej próbki
    for i in range(window_size, len(data)):
        threshold = (np.median(data[i - window_size:i]) + np.percentile(data[i - window_size:i], 1)) / 2
        thresholds.append(threshold)

    EYE_CLOSED_COUNTER = 0
    eye_counter = 0
    for i, ratio in enumerate(data):
        if ratio < thresholds[i]:
            EYE_CLOSED_COUNTER += 1
        else:
            if EYE_CLOSED_COUNTER > 0:
                eye_counter += 1
            EYE_CLOSED_COUNTER = 0
    FPS = 30
    time_from_fps = np.arange(0, len(data)) / FPS / 60
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))

    d = scipy.signal.medfilt(data, 3)
    plt.ioff()
    plt.plot(time_from_fps, d)
    plt.plot(time_from_fps, thresholds)
    plt.xlabel("Czas [min]")
    plt.ylabel("EAR")
    plt.show()
    print(eye_counter)