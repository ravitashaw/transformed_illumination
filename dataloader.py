import os
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        image = self.data_x[index]
        label = self.data_y[index]

        return image, label

    def __len__(self):
        return len(self.data_y)


def create_loaders(train_x_path, train_y_path, test_x_path, test_y_path, batch_size, val_frac=0.15):
    data_x_train = torch.from_numpy(np.load(train_x_path).swapaxes(-1, -2)).float()
    data_x_test = torch.from_numpy(np.load(test_x_path).swapaxes(-1, -2)).float()
    data_y_train = torch.from_numpy(np.load(train_y_path)).long()
    data_y_test = torch.from_numpy(np.load(test_y_path)).long()

    train_dataset = CustomDataset(data_x_train, data_y_train)
    test_dataset = CustomDataset(data_x_test, data_y_test)
    indices = np.arange(len(data_x_train))
    np.random.seed(0)
    np.random.shuffle(indices)
    val_count = int(len(indices) * val_frac)
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


def get_mnist_loaders(batch_size):
    base_path = '/home/col/data/fourier_mnist'
    train_image_path = os.path.join(base_path, "train_data_norm_v2.npy")
    train_label_path = os.path.join(base_path, "train_labels.npy")

    test_image_path = os.path.join(base_path, "test_images_v2_fixed_norm.npy")
    test_label_path = os.path.join(base_path, "test_labels.npy")

    channels = 25
    classes = 10
    train_loader, val_loader, test_loader = create_loaders(train_x_path=train_image_path, train_y_path=train_label_path,
                                                           test_x_path=test_image_path, test_y_path=test_label_path,
                                                           batch_size=batch_size)

    return train_loader, val_loader, None, channels, classes


def get_fashion_loaders(batch_size):
    base_path = r'C:\Users\clvco\Downloads\367956_717418_bundle_archive'
    if not os.path.exists(base_path):
        base_path = '/home/col/data/fourier_fashion'

    test_image_path = os.path.join(base_path, "fashion_test_norm.npy")
    test_label_path = os.path.join(base_path, "fashion_test_labels.npy")
    train_image_path = os.path.join(base_path, "fashion_train_norm.npy")
    train_label_path = os.path.join(base_path, "fashion_train_labels.npy")

    channels = 25
    classes = 10
    train_loader, val_loader, test_loader = create_loaders(train_x_path=train_image_path, train_y_path=train_label_path,
                                                           test_x_path=test_image_path, test_y_path=test_label_path,
                                                           batch_size=batch_size)

    return train_loader, val_loader, None, channels, classes
