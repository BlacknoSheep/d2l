import torchvision as tv
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image


class DataManager:
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name.lower()
        self.dataset_path = dataset_path

        if self.dataset_name == "mnist":
            self.transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,)),
            ])
            self.train_set = tv.datasets.MNIST(root=self.dataset_path,
                                               train=True, transform=self.transform, download=True)
            self.test_set = tv.datasets.MNIST(root=self.dataset_path,
                                              train=False, transform=self.transform, download=True)
        elif self.dataset_name == "fashionmnist":
            self.transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.2860,), (0.1246,)),
            ])
            self.train_set = tv.datasets.FashionMNIST(root=self.dataset_path,
                                                      train=True, transform=self.transform, download=True)
            self.test_set = tv.datasets.FashionMNIST(root=self.dataset_path,
                                                     train=False, transform=self.transform, download=True)
        elif self.dataset_name == "usps":
            self.transform = tv.transforms.Compose([
                tv.transforms.Resize(28),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5,), (0.5,)),
            ])
            self.train_set = tv.datasets.USPS(root=self.dataset_path,
                                              train=True, transform=self.transform, download=True)
            self.test_set = tv.datasets.USPS(root=self.dataset_path,
                                             train=False, transform=self.transform, download=True)
        else:
            raise ValueError("Unavailable dataset!")

        self.valid_set = None
        self.split_train_valid(valid_rate=0.2)

    def get_dataset(self, mode="train"):
        if mode == "train":
            return self.train_set
        elif mode == "valid":
            return self.valid_set
        elif mode == "test":
            return self.test_set
        else:
            raise ValueError("mode must be 'train' or 'test'!")

    def get_dataloader(self, mode="train", batch_size=64):
        if mode == "train":
            trainset = self.get_dataset(mode="train")
            return DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        elif mode == "valid":
            validset = self.get_dataset(mode="valid")
            return DataLoader(dataset=validset, batch_size=batch_size, shuffle=False)
        elif mode == "test":
            testset = self.get_dataset(mode="test")
            return DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError("mode must be 'train' or 'test'!")

    def split_train_valid(self, valid_rate=0.2):
        num_train = len(self.train_set)
        indices = list(range(num_train))
        split = int(np.floor(valid_rate * num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        valid_data = np.array([self.train_set.data[i] for i in valid_idx])
        valid_targets = np.array([self.train_set.targets[i] for i in valid_idx])
        self.valid_set = DummyDataset(valid_data, valid_targets, transform=self.transform)

        train_data = np.array([self.train_set.data[i] for i in train_idx])
        train_targets = np.array([self.train_set.targets[i] for i in train_idx])
        self.train_set = DummyDataset(train_data, train_targets, transform=self.transform)


class DummyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))
        return img, self.targets[index]

    def __len__(self):
        return len(self.data)
