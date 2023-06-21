import torchvision as tv
from torch.utils.data import DataLoader


class DataManager:
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name.lower()
        self.dataset_path = dataset_path

        if self.dataset_name == "mnist":
            self.transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,)),
            ])
            self.train_data = tv.datasets.MNIST(root=self.dataset_path,
                                                train=True, transform=self.transform, download=True)
            self.test_data = tv.datasets.MNIST(root=self.dataset_path,
                                               train=False, transform=self.transform, download=True)
        elif self.dataset_name == "fashionmnist":
            self.transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.2860,), (0.1246,)),
            ])
            self.train_data = tv.datasets.FashionMNIST(root=self.dataset_path,
                                                       train=True, transform=self.transform, download=True)
            self.test_data = tv.datasets.FashionMNIST(root=self.dataset_path,
                                                      train=False, transform=self.transform, download=True)
        elif self.dataset_name == "usps":
            self.transform = tv.transforms.Compose([
                tv.transforms.Resize(28),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5,), (0.5,)),
            ])
            self.train_data = tv.datasets.USPS(root=self.dataset_path,
                                                       train=True, transform=self.transform, download=True)
            self.test_data = tv.datasets.USPS(root=self.dataset_path,
                                                      train=False, transform=self.transform, download=True)
        else:
            raise ValueError("Unavailable dataset!")

    def get_dataset(self, mode="train"):
        if mode == "train":
            return self.train_data
        elif mode == "test":
            return self.test_data
        else:
            raise ValueError("mode must be 'train' or 'test'!")

    def get_dataloader(self, mode="train", batch_size=64):
        if mode == "train":
            trainset = self.get_dataset(mode="train")
            return DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        elif mode == "test":
            testset = self.get_dataset(mode="test")
            return DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError("mode must be 'train' or 'test'!")
