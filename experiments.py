from io import BytesIO
from os.path import join, exists
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from PIL.Image import NEAREST
from torch import nn
from torch.utils.data import Subset
from torch.utils.data.sampler import SequentialSampler, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.models import resnext101_32x8d
from torchvision.transforms import functional as tF

try:
    from pytorch_resnet_cifar10.resnet import resnet20
except ModuleNotFoundError:
    from git import Repo

    Repo.clone_from("https://github.com/akamaster/pytorch_resnet_cifar10", "./pytorch_resnet_cifar10")
    from pytorch_resnet_cifar10.resnet import resnet20

ImageFile.LOAD_TRUNCATED_IMAGES = True

from os import name as OS

NUM_WORKERS = 0 if OS == "nt" else 10


def split_data(val_split, init_train_set, transform_train, transform_test, batch_size):
    train_set = init_train_set(transform_train)
    if val_split is None:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = None
    else:
        val_set = init_train_set(transform_test)
        num_train = len(train_set)
        indices = range(num_train)
        split = int(np.floor(val_split * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(val_set, val_idx)
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
    return train_loader, val_loader


def split_data_undersample(val_split, init_train_set, transform_train, transform_test, batch_size):
    train_set = init_train_set(transform_train)
    classes = np.unique(train_set.targets)
    if val_split is None:
        samples_per_class = []
        targets = np.array(train_set.targets)
        for class_i in classes:
            samples_per_class.append((targets == class_i).sum())
        class_weights = 1 - np.array(samples_per_class) / len(train_set)
        weights = np.ones_like(targets, dtype=float)
        for i, class_i in enumerate(classes):
            idx = targets == class_i
            weights[idx] *= class_weights[i]
        sampler = WeightedRandomSampler(weights, int(min(samples_per_class)), replacement=False)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, pin_memory=True, sampler=sampler, num_workers=NUM_WORKERS)
        val_loader = None
    else:
        val_set = init_train_set(transform_test)
        num_train = len(train_set)
        indices = range(num_train)
        split = int(np.floor(val_split * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(val_set, val_idx)

        samples_per_class = []
        targets = np.array(train_set.targets)[train_idx]
        for class_i in classes:
            samples_per_class.append((targets == class_i).sum())
        class_weights = 1 - np.array(samples_per_class) / len(train_idx)
        weights = np.ones_like(targets, dtype=float)
        for i, class_i in enumerate(classes):
            idx = targets == class_i
            weights[idx] *= class_weights[i]
        sampler = WeightedRandomSampler(weights, int(min(samples_per_class)), replacement=False)

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
    return train_loader, val_loader


def get_thumbnail_transform(widths, heights):
    def add_thumbnail(img):
        thumbs = []
        for width, height in zip(widths, heights):
            width = img.width if width == -1 else width
            height = img.height if height == -1 else height
            thumb = img.copy()
            thumb.thumbnail((width, height), NEAREST)
            thumbs.append(thumb)
        return thumbs

    return add_thumbnail


def get_jpeg_transform(quality_levels):
    def add_jpgs(img):
        jpegs = []
        for quality in quality_levels:
            with BytesIO() as f:
                img.save(f, format='JPEG', quality=quality)
                f.seek(0)
                ima_jpg = Image.open(f)
                jpegs.append(ima_jpg.copy())

        return jpegs

    return add_jpgs


def get_tensor_featuremaps_transform(widths, heights, quality_levels, mean, std):
    add_thumbs = get_thumbnail_transform(widths, heights)
    n_feature_maps = len(quality_levels) + 1
    mean = mean * n_feature_maps
    std = std * n_feature_maps
    if len(quality_levels) > 0:
        get_jpeg = get_jpeg_transform(quality_levels)

    def add_feature_maps(img):
        thumbs = [img] + add_thumbs(img)
        thumbs_tensor = []
        for i, thumb in enumerate(thumbs):
            if len(quality_levels) > 0:
                fmaps = get_jpeg(thumb)
            else:
                fmaps = []
            if i == 0:
                orig_size = thumb.size
            else:
                fmaps = [fmap.resize(orig_size, NEAREST) for fmap in fmaps]
                thumb = thumb.resize(orig_size, NEAREST)
            fmaps = [tF.to_tensor(fmap) for fmap in fmaps]
            fmaps = torch.cat([tF.to_tensor(thumb)] + fmaps, 0)
            tF.normalize(fmaps, mean, std, inplace=True)
            fmaps = fmaps.view(n_feature_maps, -1, *orig_size).squeeze(0)
            thumbs_tensor.append(fmaps)
        thumbs_tensor = torch.stack(thumbs_tensor, 0).squeeze(0)
        return thumbs_tensor

    return add_feature_maps


def get_MNIST(
        batch_size=128, quality_levels=None,
        thumb_w=None, thumb_h=None, val_split=None):
    _defaults = {
        "name": "mnist",
        "quality_levels": [],
        "thumb_h": [],
        "thumb_w": [],
        "log_iter": 140,
        "lambda_patience": 1,
        "mean": [0.1307],
        "std": [0.3081],
        "loss_f": nn.CrossEntropyLoss(),
        "n_epochs": 50,
        "channel": 1,
        "n_classes": 10,
        "height": 28,
        "width": 28,
    }
    if quality_levels is None:
        quality_levels = _defaults["quality_levels"]
    if thumb_h is None:
        thumb_h = _defaults["thumb_h"]
    if thumb_w is None:
        thumb_w = _defaults["thumb_w"]

    n_quality_levels = len(quality_levels)
    channel, height, width = _defaults["channel"], _defaults["width"], _defaults["height"]

    mean = _defaults["mean"]
    std = _defaults["std"]
    featuremaps = get_tensor_featuremaps_transform(thumb_w, thumb_h, quality_levels, mean, std)

    transform_train = [featuremaps]
    transform_test = [featuremaps]

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    def init_train_set(transform):
        return torchvision.datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )

    train_loader, val_loader = split_data(
        val_split, init_train_set, transform_train, transform_test, batch_size
    )

    test_set = torchvision.datasets.MNIST('./data', train=False, download=True,
                                          transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    experiment_params = _defaults.copy()
    experiment_params.update({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "network": Net,
        "n_subpixel": channel * width * height,
        "thumb_h": thumb_h,
        "thumb_w": thumb_w,
        "n_thumbs": len(thumb_h),
        "quality_levels": quality_levels,
        "n_quality_levels": n_quality_levels,
        "n_featuremaps": 1 + n_quality_levels,
    })
    return experiment_params, _defaults


def get_fashion_MNIST(
        batch_size=128, quality_levels=None,
        thumb_w=None, thumb_h=None, val_split=None):
    _defaults = {
        "name": "fashion_mnist",
        "quality_levels": [],
        "thumb_h": [],
        "thumb_w": [],
        "log_iter": 140,
        "lambda_patience": 1,
        "mean": [0.1307],
        "std": [0.3081],
        "loss_f": nn.CrossEntropyLoss(),
        "n_epochs": 50,
        "channel": 1,
        "n_classes": 10,
        "height": 28,
        "width": 28,
    }
    if quality_levels is None:
        quality_levels = _defaults["quality_levels"]
    if thumb_h is None:
        thumb_h = _defaults["thumb_h"]
    if thumb_w is None:
        thumb_w = _defaults["thumb_w"]

    n_quality_levels = len(quality_levels)
    channel, height, width = _defaults["channel"], _defaults["width"], _defaults["height"]

    mean = _defaults["mean"]
    std = _defaults["std"]
    featuremaps = get_tensor_featuremaps_transform(thumb_w, thumb_h, quality_levels, mean, std)

    transform_train = [featuremaps]
    transform_test = [featuremaps]

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    def init_train_set(transform):
        return torchvision.datasets.FashionMNIST(
            './data', train=True, download=True, transform=transform
        )

    train_loader, val_loader = split_data(
        val_split, init_train_set, transform_train, transform_test, batch_size
    )

    test_set = torchvision.datasets.FashionMNIST('./data', train=False, download=True,
                                                 transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    experiment_params = _defaults.copy()
    experiment_params.update({

        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "network": Net,
        "n_subpixel": channel * width * height,
        "thumb_h": thumb_h,
        "thumb_w": thumb_w,
        "n_thumbs": len(thumb_h),
        "quality_levels": quality_levels,
        "n_quality_levels": n_quality_levels,
        "n_featuremaps": 1 + n_quality_levels
    })
    return experiment_params, _defaults

class Galaxy10Dataset(torch.utils.data.Dataset):

    def __init__(self, mean, std, train=True):
        from h5py import File
        url = "http://astro.utoronto.ca/~bovy/Galaxy10/Galaxy10.h5"
        file = "data/Galaxy10.h5"
        if not exists(file):
            print("first use: downloading dataset now...")
            urlretrieve(url, file)

        with File(file, "r") as f:
            images = torch.tensor(np.array(f["images"]))
            labels = torch.tensor(np.array(f["ans"])).long()

        idx = torch.arange(len(labels))
        split = int(0.9 * len(labels))
        if train:
            idx = idx[:split]
        else:
            idx = idx[split:]
        self.X = images[idx].transpose(1, 3)
        self.Y = labels[idx]

        # normalize
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.X = ((self.X / 255.) - mean) / std

        self.train = train

    def __getitem__(self, i):
        x = self.X[i]
        # x = self.transforms(x)
        # augment data while training (flip horizontal and vertically)
        if self.train:
            if np.random.rand() > .5:
                x = x.flip(1)
            if np.random.rand() > .5:
                x = x.flip(2)
            n_pixel = 8
            horizontal_crop = np.random.randint(n_pixel + 1) - np.random.randint(n_pixel + 1)

            l_pad = r_pad = t_pad = b_pad = 0
            if horizontal_crop > 0:
                x = x[:, horizontal_crop:]
                l_pad = horizontal_crop
            elif horizontal_crop < 0:
                x = x[:, :horizontal_crop]
                r_pad = -horizontal_crop
            vertical_crop = np.random.randint(n_pixel + 1) - np.random.randint(n_pixel + 1)
            if vertical_crop > 0:
                x = x[:, :, vertical_crop:]
                t_pad = vertical_crop
            elif vertical_crop < 0:
                x = x[:, :, :vertical_crop]
                b_pad = -vertical_crop

            x = F.pad(x, (t_pad, b_pad, l_pad, r_pad))

        y = self.Y[i]

        return x, y

    def __len__(self):
        return len(self.X)


def get_galaxy10(batch_size=128, quality_levels=None,
                 thumb_w=None, thumb_h=None, val_split=None):
    _defaults = {
        "name": "galaxy10",
        "quality_levels": [],
        "thumb_h": [],
        "thumb_w": [],
        "log_iter": 140,
        "lambda_patience": 1,
        "loss_f": nn.CrossEntropyLoss(),
        "mean": [0.04705882, 0.04313726, 0.02745098],
        "std": [0.06896, 0.06034, 0.0431],
        "n_epochs": 100,
        "channel": 3,
        "n_classes": 10,
        "height": 69,
        "width": 69,
    }

    if quality_levels is None:
        quality_levels = _defaults["quality_levels"]
    if thumb_h is None:
        thumb_h = _defaults["thumb_h"]
    if thumb_w is None:
        thumb_w = _defaults["thumb_w"]

    n_quality_levels = len(quality_levels)
    channel, height, width = _defaults["channel"], _defaults["width"], _defaults["height"]

    mean = _defaults["mean"]
    std = _defaults["std"]

    def init_train_set(transform):
        return Galaxy10Dataset(mean, std, train=True)

    train_loader, val_loader = split_data(
        val_split, init_train_set, None, None, batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        Galaxy10Dataset(mean, std, train=False), batch_size=batch_size,
        pin_memory=True, shuffle=False, num_workers=NUM_WORKERS
    )

    def get_net():
        net = resnet20()
        net.layer1[0].stride = 2
        net.linear.reset_parameters()
        net = torch.nn.DataParallel(net)
        return net

    experiment_params = _defaults.copy()
    experiment_params.update({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "network": get_net,
        "channel": channel,
        "width": width,
        "height": height,
        "n_subpixel": channel * width * height,
        "thumb_h": thumb_h,
        "thumb_w": thumb_w,
        "n_thumbs": len(thumb_h),
        "mean": mean,
        "std": std,
        "quality_levels": quality_levels,
        "n_quality_levels": n_quality_levels,
        "n_featuremaps": 1 + n_quality_levels,
    })

    return experiment_params, _defaults


def get_cifar10(
        batch_size=128, quality_levels=None,
        thumb_w=None, thumb_h=None, val_split=None):
    _defaults = {
        "name": "cifar10",
        "quality_levels": [],
        "thumb_h": [],
        "thumb_w": [],
        "log_iter": 140,
        "lambda_patience": 1,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "loss_f": nn.CrossEntropyLoss(),
        "n_epochs": 100,
        "channel": 3,
        "n_classes": 10,
        "height": 32,
        "width": 32,
    }
    if quality_levels is None:
        quality_levels = _defaults["quality_levels"]
    if thumb_h is None:
        thumb_h = _defaults["thumb_h"]
    if thumb_w is None:
        thumb_w = _defaults["thumb_w"]

    n_quality_levels = len(quality_levels)
    channel, height, width = _defaults["channel"], _defaults["width"], _defaults["height"]

    mean = _defaults["mean"]
    std = _defaults["std"]
    featuremaps = get_tensor_featuremaps_transform(thumb_w, thumb_h, quality_levels, mean, std)

    transform_train = [
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4, padding_mode="reflect"),
        featuremaps
    ]
    transform_test = [featuremaps]

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    def init_train_set(transform):
        return torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )

    train_loader, val_loader = split_data(
        val_split, init_train_set, transform_train, transform_test, batch_size
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS)

    def get_net():
        net = resnet20()
        net = torch.nn.DataParallel(net)
        # save = torch.load("./pytorch_resnet_cifar10/pretrained_models/resnet20.th", map_location="cpu")
        # net.load_state_dict(save["state_dict"])
        return net

    experiment_params = _defaults.copy()
    experiment_params.update({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "network": get_net,
        "n_subpixel": channel * width * height,
        "thumb_h": thumb_h,
        "thumb_w": thumb_w,
        "n_thumbs": len(thumb_h),
        "quality_levels": quality_levels,
        "n_quality_levels": n_quality_levels,
        "n_featuremaps": 1 + n_quality_levels
    })

    return experiment_params, _defaults


def get_svhn(
        batch_size=128, quality_levels=None,
        thumb_w=None, thumb_h=None, val_split=None):
    _defaults = {
        "name": "svhn",
        "quality_levels": [],
        "thumb_h": [],
        "thumb_w": [],
        "log_iter": 140,
        "lambda_patience": 1,
        "loss_f": nn.CrossEntropyLoss(),
        "mean": [0.5] * 3,
        "std": [0.5] * 3,
        "n_epochs": 100,
        "channel": 3,
        "n_classes": 10,
        "height": 32,
        "width": 32,
    }
    if quality_levels is None:
        quality_levels = _defaults["quality_levels"]
    if thumb_h is None:
        thumb_h = _defaults["thumb_h"]
    if thumb_w is None:
        thumb_w = _defaults["thumb_w"]

    n_quality_levels = len(quality_levels)
    channel, height, width = _defaults["channel"], _defaults["width"], _defaults["height"]

    mean = _defaults["mean"]
    std = _defaults["std"]
    featuremaps = get_tensor_featuremaps_transform(thumb_w, thumb_h, quality_levels, mean, std)

    transform_train = [
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4, padding_mode="reflect"),
        featuremaps
    ]
    transform_test = [featuremaps]

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    def init_train_set(transform):
        return torchvision.datasets.SVHN(
            root='./data', split="train", download=True, transform=transform
        )

    train_loader, val_loader = split_data(
        val_split, init_train_set, transform_train, transform_test, batch_size
    )

    test_set = torchvision.datasets.SVHN(
        root='./data', split="test", download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS)

    def get_net():
        net = resnet20()
        net.linear.reset_parameters()
        net = torch.nn.DataParallel(net)
        return net

    experiment_params = _defaults.copy()
    experiment_params.update({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "network": get_net,
        "channel": channel,
        "width": width,
        "height": height,
        "n_subpixel": channel * width * height,
        "thumb_h": thumb_h,
        "thumb_w": thumb_w,
        "n_thumbs": len(thumb_h),
        "mean": mean,
        "std": std,
        "quality_levels": quality_levels,
        "n_quality_levels": n_quality_levels,
        "n_featuremaps": 1 + n_quality_levels,
    })

    return experiment_params, _defaults

def get_remotesensing(
        batch_size=128, quality_levels=None,
        thumb_w=None, thumb_h=None, val_split=None):
    _defaults = {
        "name": "remotesensing",
        "quality_levels": [],
        "thumb_h": [],
        "thumb_w": [],
        "log_iter": 140,
        "lambda_patience": 1,
        "loss_f": nn.CrossEntropyLoss(),
        "mean": [
            56.63172, 24.977304, 24.998138, 52.370487, 65.14917,
            27.397463, 62.00862, 22.310347, 22.100933, 71.31217,
            64.60853, 23.660484, 45.66891, 18.37689, 21.051277,
            36.00949, 48.651344, 22.681187, 52.068127, 22.519337,
            24.185656, 63.286736, 72.094696, 31.067451, 70.99113,
            46.457024, 43.007847, 104.74432, 80.80541, 47.81025,
            40.194836, 19.74167, 20.012398, 35.994465, 62.428997,
            26.348455
        ],
        "std": [
            56.63172, 24.977304, 24.998138, 52.370487, 65.14917,
            27.397463, 62.00862, 22.310347, 22.100933, 71.31217,
            64.60853, 23.660484, 45.66891, 18.37689, 21.051277,
            36.00949, 48.651344, 22.681187, 52.068127, 22.519337,
            24.185656, 63.286736, 72.094696, 31.067451, 70.99113,
            46.457024, 43.007847, 104.74432, 80.80541, 47.81025,
            40.194836, 19.74167, 20.012398, 35.994465, 62.428997,
            26.348455
        ],
        "n_epochs": 200,
        "channel": 36,
        "n_classes": 13,
        "height": 35,
        "width": 35,
    }
    error = "Not supported for this dataset, because we would need to create an Image, which looses precision"
    if quality_levels is None:
        quality_levels = _defaults["quality_levels"]
    else:
        raise Exception(error)
    if thumb_h is None:
        thumb_h = _defaults["thumb_h"]
    else:
        raise Exception(error)
    if thumb_w is None:
        thumb_w = _defaults["thumb_w"]
    else:
        raise Exception(error)

    n_quality_levels = len(quality_levels)
    channel, height, width = _defaults["channel"], _defaults["width"], _defaults["height"]

    mean = _defaults["mean"]
    std = _defaults["std"]

    transform_train = [
        transforms.ToPILImage(),
        transforms.RandomCrop(35, 5, padding_mode="reflect"),
        transforms.ToTensor()
    ]
    transform_test = []

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    datapath = "data/remote_sensing_change"

    def init_train_set(transforms):
        return RemoteSenseDataset(datapath, transforms, mean, std, train=True)

    train_loader, val_loader = split_data(
        val_split, init_train_set, transform_train, transform_test, batch_size
    )

    test_set = RemoteSenseDataset(datapath, transform_test, mean, std, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS)

    class Net(nn.Module):
        def __init__(self, channel):
            super(Net, self).__init__()
            self.n_classes = 13
            self.pad = nn.ReflectionPad2d(1)
            self.conv1 = nn.Conv2d(channel, 64, 3, 2)
            self.conv12 = nn.Conv2d(64, 64, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 2)
            self.conv22 = nn.Conv2d(128, 128, 1)
            self.conv3 = nn.Conv2d(128, 256, 3, 2)
            self.conv32 = nn.Conv2d(256, 512, 1)
            self.convo = nn.Conv2d(512, self.n_classes, 1)

        def forward(self, x):
            x = F.leaky_relu(
                self.conv1(self.pad(x))
            )
            x = F.leaky_relu(
                self.conv12(self.pad(x))
            )
            x = F.leaky_relu(
                self.conv2(self.pad(x))
            )
            x = F.leaky_relu(
                self.conv22(self.pad(x))
            )
            x = F.leaky_relu(
                self.conv3(self.pad(x))
            )
            x = F.leaky_relu(
                self.conv32(self.pad(x))
            )
            x = self.convo(x)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.n_classes)

            return x

    def get_net():
        net = Net(channel)
        return net

    experiment_params = _defaults.copy()
    experiment_params.update({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "network": get_net,
        "channel": channel,
        "width": width,
        "height": height,
        "n_subpixel": channel * width * height,
        "thumb_h": thumb_h,
        "thumb_w": thumb_w,
        "n_thumbs": len(thumb_h),
        "mean": mean,
        "std": std,
        "quality_levels": quality_levels,
        "n_quality_levels": n_quality_levels,
        "n_featuremaps": 1 + n_quality_levels
    })

    return experiment_params, _defaults


class RemoteSenseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms, mean, std, train):
        self.data_path = data_path

        # only get length
        self.X = np.load(join(data_path, 'X_train.npy'), mmap_mode='r')
        self.Y = np.load(join(self.data_path, 'y_train.npy'), mmap_mode='r')
        self.len_X = len(self.X)
        self.index = np.arange(self.len_X)
        rs = np.random.RandomState(42)
        rs.shuffle(self.index)
        self.split = int(0.5 * self.len_X)

        self.mean = np.array(mean, dtype=np.float32)[None, None, :]
        self.std = np.array(std, dtype=np.float32)[None, None, :]

        self.train = train

        self.transforms = transforms

    def __getitem__(self, i):
        if self.train:
            index = self.index[i]
        else:
            index = self.index[self.split + i]

        x = self.X[index]
        y = self.Y[index]

        x = (x - self.mean) / self.std
        x = torch.tensor(x).permute(2, 0, 1)
        # x = self.transforms(x)
        # augment data while training (flip horizontal and vertically)
        if self.train:
            if np.random.rand() > .5:
                x = x.flip(1)
            if np.random.rand() > .5:
                x = x.flip(2)

        y = torch.tensor(y).long()

        return x, y

    def __len__(self):
        if self.train:
            return self.split
        else:
            return self.len_X - self.split


class ShipDataset(torch.utils.data.Dataset):
    def __init__(self, data, cache_transform, transform):
        self.data = data
        self.targets = [img[1] for img in data.imgs]
        self.cache_transform = cache_transform
        self.transform = transform
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img = self.cache[idx]
        except KeyError:
            img = self.cache[idx] = self.cache_transform(self.data[idx][0])

        return self.transform(img), self.targets[idx]


def get_ships(
        batch_size=128, quality_levels=None,
        thumb_w=None, thumb_h=None, val_split=None):
    _defaults = {
        "name": "ships",
        "quality_levels": [],
        "thumb_h": [],
        "thumb_w": [],
        "log_iter": 140,
        "lambda_patience": 1,
        "loss_f": nn.CrossEntropyLoss(),
        "mean": [0.20182742, 0.28122104, 0.31776145],
        "std": [0.2094284, 0.19005547, 0.16852541],
        "n_epochs": 200,
        "channel": 3,
        "n_classes": 2,
        "height": 768 // 2,  # orig is 768
        "width": 768 // 2,  # orig is 768
    }
    if quality_levels is None:
        quality_levels = _defaults["quality_levels"]
    if thumb_h is None:
        thumb_h = _defaults["thumb_h"]
    if thumb_w is None:
        thumb_w = _defaults["thumb_w"]

    n_quality_levels = len(quality_levels)
    n_classes = _defaults["n_classes"]
    channel, height, width = _defaults["channel"], _defaults["width"], _defaults["height"]

    imbalance_minority = 34052 / (141496 + 34052)

    mean = _defaults["mean"]
    std = _defaults["std"]
    featuremaps = get_tensor_featuremaps_transform(thumb_w, thumb_h, quality_levels, mean, std)

    cache_transform = transforms.Compose([
        transforms.Resize(_defaults["height"])
    ])

    transform_train = [
        transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(_defaults["height"], 10, padding_mode="reflect"),
        featuremaps
    ]
    transform_test = [featuremaps]


    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    datapath = "./data/ship_classification"

    def init_train_set(transforms):
        data = ImageFolder(f"{datapath}/train", )
        return ShipDataset(data, cache_transform, transforms)

    train_loader, val_loader = split_data_undersample(
        val_split, init_train_set, transform_train, transform_test, batch_size
    )

    test_data = ImageFolder(f"{datapath}/test")
    test_set = ShipDataset(test_data, cache_transform, transform_test)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS)

    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def get_net():
        net = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_1', pretrained=True)

        classifier = nn.Conv2d(512, n_classes, kernel_size=1)
        classifier.bias = nn.Parameter(net.classifier[1].bias[:n_classes])
        classifier.weight = nn.Parameter(net.classifier[1].weight[:n_classes])
        net.classifier[1] = classifier

        nn.init.normal_(net.classifier[1].weight, mean=0.0, std=0.01)
        net.features[-1].apply(init_weights)

        net = torch.nn.DataParallel(net)

        return net

    experiment_params = _defaults.copy()
    experiment_params.update({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "network": get_net,
        "channel": channel,
        "width": width,
        "height": height,
        "n_subpixel": channel * width * height,
        "thumb_h": thumb_h,
        "thumb_w": thumb_w,
        "n_thumbs": len(thumb_h),
        "mean": mean,
        "std": std,
        "quality_levels": quality_levels,
        "n_quality_levels": n_quality_levels,
        "n_featuremaps": 1 + n_quality_levels,
    })

    return experiment_params, _defaults
