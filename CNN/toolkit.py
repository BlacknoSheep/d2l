import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics


def show_result(data: dict, title, save_path=None):
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
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()


def show_pr(logits, labels, title, save_path=None):
    # print("logits.shape: ", logits.shape)  # (n, num_classes)
    # print("labels.shape: ", labels.shape)  # (n, )

    # 类别数
    num_classes = logits.shape[1]

    # 绘制每个类别的pr曲线
    plt.figure(title, figsize=(10, 5))
    for i in range(num_classes):
        # 计算每个类别的pr曲线
        precision, recall, thresholds = metrics.precision_recall_curve(labels, logits[:, i], pos_label=i)

        # 绘制pr曲线
        plt.plot(recall, precision, label="class {}".format(i))

    plt.title(title)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
