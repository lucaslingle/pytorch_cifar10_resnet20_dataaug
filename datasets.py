import torch as tc
import torchvision as tv
import numpy as np
import PIL


class CIFAR10HePreprocessing(tc.utils.data.Dataset):
    # CIFAR-10 with preprocessing as described in Section 4.2 of He et al., 2015.
    def __init__(self, root, train):
        self.root = root
        self.train = train
        self.dataset = tv.datasets.CIFAR10(
            root=root, train=train, download=True, transform=None, target_transform=None)
        self.transform = self.get_transform(train)
        self.target_transform = None
        self.per_pixel_means = self.get_per_pixel_mean()

    def get_per_pixel_mean(self):
        training_data = tv.datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=None)

        X, y = training_data[0]
        X = np.array(X)
        per_pixel_means = np.zeros(dtype=np.float32, shape=X.shape)

        for i in range(0, len(training_data)):
            X, y = training_data[i]
            X = np.array(X)
            per_pixel_means += X

        per_pixel_means = per_pixel_means / float(len(training_data))
        return per_pixel_means

    def get_transform(self, train):
        if train:
            transform = tv.transforms.Compose([
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomCrop(size=(32, 32), padding=(4, 4)),
                tv.transforms.ToTensor()
            ])
            return transform
        else:
            return tv.transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        X = np.array(X, dtype=np.float32)
        npimage = np.round(X - self.per_pixel_means).astype(np.uint8) # cant convert back to PIL Image without this O_o
        image = PIL.Image.fromarray(npimage)
        label = y
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
