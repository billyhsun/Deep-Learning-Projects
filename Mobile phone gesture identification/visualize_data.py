'''
    Visualize some samples.
'''

import numpy as np
from matplotlib import pyplot as plt


def visualize_data(array):
    times = []
    for i in range(100):
        times.append(i*20)
    times = np.array(times)

    data = []
    data.append(array[0][4])    # Letter a
    data.append(array[0][15])
    data.append(array[0][42])
    data.append(array[1][24])   # Letter b
    data.append(array[1][47])
    data.append(array[1][134])

    for i in range(6):
        d = np.transpose(data[i])
        plt.figure(figsize=(10, 6))
        for j in range(6):
            plt.plot(times, d[j])
        plt.show()
    return
