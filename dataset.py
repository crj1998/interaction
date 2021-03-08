
import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def createCIFARmini(root="./data/", train=True, num_per_class=500):
    """Create mini CIFAR10

    Fisrt, download torchvision.datasets.CIFAR10's raw data, then parse raw CIFAR10 file and extract part of data.

    Keyword Arguments:
        root {str} -- same as torchvision.datasets.CIFAR10's root (default: {"./data/"})
        train {bool} -- Train or test set (default: {True})
        num_per_class {number} -- number of images per class (default: {500})

    Returns:
        ndarray, ndarray -- images (num_per_class, 3, 32, 32), labels(num_per_class, )
    """
    assert isinstance(train, bool)
    if not os.path.exists(os.path.join(root, "cifar-10-batches-py")):
        raise RuntimeError("Download CIFAR10 first.")
    path = os.path.join(root, "cifar-10-mini")
    if not os.path.exists(path):
        os.mkdir(path)
    if train:
        save_path = os.path.join(path, f"train_batch_{num_per_class}")
    else:
        save_path = os.path.join(path, f"test_batch_{num_per_class}")
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data_dict = pickle.load(f, encoding="bytes")
        return data_dict[b'data'], data_dict[b'labels']

    if train:
        data_path = os.path.join(root, "cifar-10-batches-py/data_batch_1")
    else:
        data_path = os.path.join(root, "cifar-10-batches-py/test_batch")
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f, encoding="bytes")
    del data_dict[b'batch_label'], data_dict[b'filenames']
    labels = np.array(data_dict[b'labels'])
    images = np.reshape(data_dict[b'data'], (10000, 3, 32, 32))
    images = images.transpose((0, 2, 3, 1))
    CLASSES = 10
    index = np.zeros((CLASSES * num_per_class,), dtype=np.uint16)
    count = np.zeros((CLASSES,), dtype=np.uint16)
    num = 0
    for i, label in enumerate(labels):
        if count[label] < num_per_class:
            index[num] = i
            count[label] += 1
            num += 1
        if (count >= num_per_class).all():
            break

    data_dict[b'labels'] = labels[index]
    data_dict[b'data'] = images[index]

    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    return data_dict[b'data'], data_dict[b'labels']

class CIFAR10mini(Dataset):
    """ Load CIFAR10 mini
    Use like torchvision.datasets.CIFAR10, just add num_per_class parameter, which means how many samples used.
    """

    def __init__(self, root, train=True, num_per_class=500, transform=None):
        super(CIFAR10mini, self).__init__()
        assert isinstance(train, bool)
        self.images, self.labels = createCIFARmini(root, train, num_per_class)
        self.num_samples = self.labels.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.num_samples