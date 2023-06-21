import matplotlib.pyplot as plt
import numpy as np


def show_result(data:dict, title):
    colors = ["darkred", "red", "darkgreen", "green"]
    linestyles = ['-', ':', '-', ':']
    plt.figure(title, figsize=(10, 5))
    i = 0
    for key, val in data.items():
        plt.plot(np.arange(1, len(val) + 1), val, label=key, color=colors[i], linestyle=linestyles[i])
        i += 1
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()
    plt.show()