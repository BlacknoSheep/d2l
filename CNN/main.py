import torch
import torchsummary
import time
import os
from trainer import Trainer
from toolkit import show_result, show_pr
from logger import logger

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

model_name = "AlexNet"
dataset_name = "USPS"
dataset_path = r"E:\python\dataset\USPS"
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
saved_path = "./saved"
# 创建保存模型的文件夹
if not os.path.exists(saved_path):
    os.makedirs(saved_path)


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

    # test
    test_loss, test_accuracy, logits, y_trues = tr.test()
    print("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(test_loss, test_accuracy))

    # 保存日志
    logger.add("model: {}".format(model_name))
    logger.add("dataset: {}".format(dataset_name))
    for key, val in result.items():
        logger.add("{}: {}".format(key, val))
    logger.add("time cost: {:.2f} s".format(end - start))
    logger.add("test loss: {:.4f}, test accuracy: {:.4f}".format(test_loss, test_accuracy))
    logger.save(os.path.join(saved_path, "{} on {}.txt".format(model_name, dataset_name)))

    # 显示网络规模
    torchsummary.summary(tr.model, (1, 28, 28))

    # 显示训练结果
    show_result(result, title="{} on {}".format(model_name, dataset_name),
                save_path=os.path.join(saved_path, "{} on {}.png".format(model_name, dataset_name)))

    # 显示PR曲线
    show_pr(logits, y_trues, title="{} on {}, PR".format(model_name, dataset_name),
            save_path=os.path.join(saved_path, "{} on {} - PR.png".format(model_name, dataset_name)))
