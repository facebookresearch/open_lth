# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import concurrent
import numpy as np
import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform


def _get_samples(root, y_name, y_num):
    y_dir = os.path.join(root, y_name)
    if not get_platform().isdir(y_dir): return []
    output = [(os.path.join(y_dir, f), y_num) for f in get_platform().listdir(y_dir) if f.lower().endswith('jpeg')]
    return output


class Dataset(base.ImageDataset):
    """ImageNet"""

    def __init__(self, loc: str, image_transforms):
        # Load the data.
        classes = sorted(get_platform().listdir(loc))
        samples = []

        if get_platform().num_workers > 0:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=get_platform().num_workers)
            futures = [executor.submit(_get_samples, loc, y_name, y_num) for y_num, y_name in enumerate(classes)]
            for d in concurrent.futures.wait(futures)[0]: samples += d.result()
        else:
            for y_num, y_name in enumerate(classes):
                samples += _get_samples(loc, y_name, y_num)

        examples, labels = zip(*samples)
        super(Dataset, self).__init__(
            np.array(examples), np.array(labels), image_transforms,
            [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @staticmethod
    def num_train_examples(): return 1281167

    @staticmethod
    def num_test_examples(): return 50000

    @staticmethod
    def num_classes(): return 1000

    @staticmethod
    def _augment_transforms():
        return [
            torchvision.transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(0.8, 1.25)),
            torchvision.transforms.RandomHorizontalFlip()
        ]

    @staticmethod
    def _transforms():
        return [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224)]

    @staticmethod
    def get_train_set(use_augmentation):
        transforms = Dataset._augment_transforms() if use_augmentation else Dataset._transforms()
        return Dataset(os.path.join(get_platform().imagenet_root, 'train'), transforms)

    @staticmethod
    def get_test_set():
        return Dataset(os.path.join(get_platform().imagenet_root, 'val'), Dataset._transforms())

    @staticmethod
    def example_to_image(example):
        with get_platform().open(example, 'rb') as fp:
            return Image.open(fp).convert('RGB')


DataLoader = base.DataLoader
