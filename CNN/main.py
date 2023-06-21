import torch
import torchsummary
import time
from trainer import Trainer
from toolkit import show_result

model_settings = {
    "Linear":     [10, 0.01],
    "MLP":        [25, 0.01],
    "LeNet":      [25, 0.01],
    "LeNetReLU":  [25, 0.01],
    "AlexNet":    [15, 0.001],
    "VGG":        [15, 0.001],
    "NiN":        [15, 0.005],
    "GoogLeNet":  [15, 0.001],
    "ResNet":     [10, 0.001],
    "ResNetV2":   [10, 0.001],
    "DenseNet":   [10, 0.001],
    }

model_name = "NiN"
dataset_name = "USPS"
dataset_path = r"E:\python\dataset\USPS"
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    tr = Trainer(model_name=model_name,
                 dataset_name=dataset_name,
                 dataset_path=dataset_path,
                 batch_size=batch_size,
                 device=device)

    print("Start training...")
    start = time.time()
    result = tr.train(epochs=model_settings[model_name][0],
                      lr=model_settings[model_name][1],)
    end = time.time()
    print("Training finished. Time cost: {:.2f} s".format(end - start))

    # 显示网络规模
    torchsummary.summary(tr.model, (1, 28, 28))

    # 显示训练结果
    show_result(result, title="{} on {}".format(model_name, dataset_name))
