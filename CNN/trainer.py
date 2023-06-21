import numpy as np
from tqdm import tqdm
import torch
import models
from datamanager import DataManager


class Trainer:
    def __init__(self, model_name, dataset_name, dataset_path, batch_size, device):
        self.model = self.get_model(model_name)
        self.data_manger = DataManager(dataset_name, dataset_path)
        self.batch_size = batch_size
        self.device = device

        self.model.to(device)

        self.train_loader = self.data_manger.get_dataloader(mode="train", batch_size=self.batch_size)
        self.test_loader = self.data_manger.get_dataloader(mode="test", batch_size=self.batch_size)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = None

    def train(self, epochs=10, lr=0.001):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        result = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": []
        }
        for epoch in range(epochs):
            train_loss, train_accuracy = self._train_epoch()
            test_loss, test_accuracy = self.eval()
            print("Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(
                epoch, train_loss, train_accuracy, test_loss, test_accuracy))

            result["train_loss"].append(train_loss)
            result["train_accuracy"].append(train_accuracy)
            result["test_loss"].append(test_loss)
            result["test_accuracy"].append(test_accuracy)
        return result

    def _train_epoch(self):
        self.model.train()
        sum_loss = 0
        sum_accuracy = 0
        num_batches = len(self.train_loader)

        for batch_idx, data in enumerate(tqdm(self.train_loader)):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            predicts = self.model(inputs)
            loss = self.criterion(predicts, labels)
            loss.backward()
            self.optimizer.step()

            predicts = predicts.detach().to('cpu').numpy()
            predict_labels = np.argmax(predicts, axis=1)
            labels = labels.to('cpu').numpy()
            sum_loss += loss.item()
            sum_accuracy += np.mean(np.equal(predict_labels, labels))
        return sum_loss / num_batches, sum_accuracy / num_batches

    def eval(self):
        self.model.eval()
        sum_loss = 0
        sum_accuracy = 0
        num_batches = len(self.test_loader)

        for batch_index, data in enumerate(self.test_loader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                predicts = self.model(inputs)
                loss = self.criterion(predicts, labels)

            predicts = predicts.detach().to('cpu').numpy()
            predict_labels = np.argmax(predicts, axis=1)
            labels = labels.to('cpu').numpy()
            sum_loss += loss.item()
            sum_accuracy += np.mean(np.equal(predict_labels, labels))
        return sum_loss / num_batches, sum_accuracy / num_batches

    def get_model(self, model_name):
        if model_name == "Linear":
            return models.Linear()
        elif model_name == "MLP":
            return models.MLP()
        elif model_name == "LeNet":
            return models.LeNet()
        elif model_name == "LeNetReLU":
            return models.LeNetReLU()
        elif model_name == "AlexNet":
            return models.AlexNetSmall()
        elif model_name == "VGG":
            return models.VGG9Small()
        elif model_name == "NiN":
            return models.NiN()
        elif model_name == "GoogLeNet":
            return models.GoogLeNet()
        elif model_name == "ResNet":
            return models.ResNet()
        elif model_name == "ResNetV2":
            return models.ResNetV2()
        elif model_name == "DenseNet":
            return models.DenseNet()
        else:
            raise ValueError("Invalid model name: {}".format(model_name))
