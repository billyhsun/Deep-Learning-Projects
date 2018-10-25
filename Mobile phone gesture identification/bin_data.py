'''
    Visualize some basic statistics of our dataset.
'''

import numpy as np
from matplotlib import pyplot as plt


def compute_stats(array):
    stats = []

    data = []
    data.append(array[0][201])   # Letter a
    data.append(array[1][126])   # Letter b
    data.append(array[2][47])    # Letter c

    for i in range(3):
        d = np.transpose(data[i])
        means = []
        stddev = []
        for j in range(6):
            means.append(np.mean(d[j]))
            stddev.append(np.std(d[j]))
        ind = [1, 2, 3, 4, 5, 6]
        plt.figure(figsize=(10, 6))
        plt.bar(ind, means, yerr=stddev)
        plt.show()
    return
