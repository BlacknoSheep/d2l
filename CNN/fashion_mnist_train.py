import pandas as pd
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import torchsummary
from fashion_mnist_models import*
import sys

# sys.path.append("..")
# from my_tools import my_tools


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
# text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

transform = transforms.Compose([
    # transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.1246,))
])

train_data = datasets.FashionMNIST(root=r"E:\python\jupyter_notebook\d2l-v2\dataset",
                                   train=True, transform=transform, download=True)
test_data = datasets.FashionMNIST(root=r"E:\python\jupyter_notebook\d2l-v2\dataset",
                                  train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# models = ["Linear", "MLP", "LeNet", "LeNetReLU", "AlexNetSmall", "VGG9Small", "NiN", "GoogLeNet", "ResNet"]
# model_id = -1
# epochs_list = [10, 25, 25, 25, 15, 15, 15, 15, 15]
# lr_list = [0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.005, 0.001, 0.001]

# model_name = models[model_id]
# epochs = epochs_list[model_id]
# lr = lr_list[model_id]

model_settings = {
    "Linear":       [10, 0.01],
    "MLP":          [25, 0.01],
    "LeNet":        [25, 0.01],
    "LeNetReLU":    [25, 0.01],
    "AlexNetSmall": [15, 0.001],
    "VGG9Small":    [15, 0.001],
    "NiN":          [15, 0.005],
    "GoogLeNet":    [15, 0.001],
    "ResNet":       [10, 0.001],
    "ResNetV2":     [10, 0.001],
    "DenseNet":     [10, 0.001],
    }

model_name = "DenseNet"
epochs = model_settings[model_name][0]
lr = model_settings[model_name][1]

model = eval(model_name)().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train():
    model.train()
    # batch_loss = []
    sum_loss = 0
    sum_accuracy = 0
    num_batches = len(train_loader)

    for batch_idx, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predicts = model(inputs)
        loss = criterion(predicts, labels)
        loss.backward()
        optimizer.step()

        # batch_loss.append(loss.item())
        predicts = predicts.detach().to('cpu').numpy()
        predict_labels = np.argmax(predicts, axis=1)
        labels = labels.to('cpu').numpy()
        sum_loss += loss.item()
        sum_accuracy += np.mean(np.equal(predict_labels, labels))
    return sum_loss / num_batches, sum_accuracy / num_batches


def test():
    model.eval()
    # batch_loss = []
    sum_loss = 0
    sum_accuracy = 0
    num_batches = len(test_loader)

    for batch_index, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        predicts = model(inputs)
        loss = criterion(predicts, labels)
        # batch_loss.append(loss.item())

        predicts = predicts.detach().to('cpu').numpy()
        predict_labels = np.argmax(predicts, axis=1)
        labels = labels.to('cpu').numpy()
        sum_loss += loss.item()
        sum_accuracy += np.mean(np.equal(predict_labels, labels))
    return sum_loss / num_batches, sum_accuracy / num_batches


if __name__ == "__main__":
    log_columns = ["train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    logs = [[], [], [], []]  # 统计信息，字典的列表
    lowest_loss = float("inf")  # 最低测试集损失
    start = time.time()
    for epoch in range(epochs):
        loss, accuracy = train()
        logs[0].append(loss)
        logs[1].append(accuracy)
        print(f"epoch: {epoch + 1}, {log_columns[0]}: {loss}, {log_columns[1]}: {accuracy}", end=' ')

        loss, accuracy = test()
        logs[2].append(loss)
        logs[3].append(accuracy)
        print(f"{log_columns[2]}: {loss}, {log_columns[3]}: {accuracy}")

        # 保存test loss最低的模型
        if loss < lowest_loss:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "./models/" + model_name + ".pth")
            lowest_loss = loss
    end = time.time()

    # 显示训练时间及网络规模
    print(f"time use: {end - start} seconds")
    torchsummary.summary(model, (1, 28, 28))

    # 保存训练过程信息
    logs_numpy = np.array(logs)
    logs_numpy = logs_numpy.T
    os.makedirs("logs", exist_ok=True)
    df = pd.DataFrame(logs_numpy, columns=log_columns)
    df.index += 1
    df.to_csv("./logs/"+model_name+".csv")

    with open("./logs/"+model_name+"_stat.txt", 'w') as f:
        f.write(f"time use: {end - start} seconds")

    # 绘制损失和精度曲线
    colors = ["darkred", "red", "darkgreen", "green"]
    linestyles = ['-', ':', '-', ':']
    plt.figure(model_name, figsize=(10, 5))
    for i in range(4):
        plt.plot(np.arange(1, len(logs[i]) + 1), logs[i], label=log_columns[i], color=colors[i], linestyle=linestyles[i])
    # plt.plot(np.arange(0, len(logs[1])), logs[1], label="train_accuracy")
    # plt.plot(np.arange(0, len(logs[2])), logs[2], label="test_loss")
    # plt.plot(np.arange(0, len(logs[3])), logs[3], label="test_accuracy")
    plt.title(model_name)
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()
    plt.show()
