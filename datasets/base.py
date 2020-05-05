# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import numpy as np
from PIL import Image
import torch
import torchvision

from platforms.platform import get_platform


class Dataset(abc.ABC, torch.utils.data.Dataset):
    """The base class for all datasets in this framework."""

    @staticmethod
    @abc.abstractmethod
    def num_test_examples() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def num_train_examples() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def num_classes() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_train_set(use_augmentation: bool) -> 'Dataset':
        pass

    @staticmethod
    @abc.abstractmethod
    def get_test_set() -> 'Dataset':
        pass

    def __init__(self, examples: np.ndarray, labels):
        """Create a dataset object.

        examples is a numpy array of the examples (or the information necessary to get them).
        Only the first dimension matters for use in this abstract class.

        labels is a numpy array of the labels. Each entry is a zero-indexed integer encoding
        of the label.
        """

        if examples.shape[0] != labels.shape[0]:
            raise ValueError('Different number of examples ({}) and labels ({}).'.format(
                             examples.shape[0], examples.shape[0]))
        self._examples = examples
        self._labels = labels if isinstance(labels, np.ndarray) else labels.numpy()
        self._subsampled = False

    def randomize_labels(self, seed: int, fraction: float) -> None:
        """Randomize the labels of the specified fraction of the dataset."""

        num_to_randomize = np.ceil(len(self._labels) * fraction).astype(int)
        randomized_labels = np.random.RandomState(seed=seed).randint(self.num_classes(), size=num_to_randomize)
        examples_to_randomize = np.random.RandomState(seed=seed+1).permutation(len(self._labels))[:num_to_randomize]
        self._labels[examples_to_randomize] = randomized_labels

    def subsample(self, seed: int, fraction: float) -> None:
        """Subsample the dataset."""

        if self._subsampled:
            raise ValueError('Cannot subsample more than once.')
        self._subsampled = True

        examples_to_retain = np.ceil(len(self._labels) * fraction).astype(int)
        examples_to_retain = np.random.RandomState(seed=seed+1).permutation(len(self._labels))[:examples_to_retain]
        self._examples = self._examples[examples_to_retain]
        self._labels = self._labels[examples_to_retain]

    def __len__(self):
        return self._labels.size

    def __getitem__(self, index):
        """If there is custom logic for example loading, this method should be overridden."""

        return self._examples[index], self._labels[index]


class ImageDataset(Dataset):
    @abc.abstractmethod
    def example_to_image(self, example: np.ndarray) -> Image: pass

    def __init__(self, examples, labels, image_transforms=None, tensor_transforms=None,
                 joint_image_transforms=None, joint_tensor_transforms=None):
        super(ImageDataset, self).__init__(examples, labels)
        self._image_transforms = image_transforms or []
        self._tensor_transforms = tensor_transforms or []
        self._joint_image_transforms = joint_image_transforms or []
        self._joint_tensor_transforms = joint_tensor_transforms or []

        self._composed = None

    def __getitem__(self, index):
        if not self._composed:
            self._composed = torchvision.transforms.Compose(
                self._image_transforms + [torchvision.transforms.ToTensor()] + self._tensor_transforms)

        example, label = self._examples[index], self._labels[index]
        example = self.example_to_image(example)
        for t in self._joint_image_transforms: example, label = t(example, label)
        example = self._composed(example)
        for t in self._joint_tensor_transforms: example, label = t(example, label)
        return example, label

    def blur(self, blur_factor: float) -> None:
        """Add a transformation that blurs the image by downsampling by blur_factor."""

        def blur_transform(image):
            size = list(image.size)
            image = torchvision.transforms.Resize([int(s / blur_factor) for s in size])(image)
            image = torchvision.transforms.Resize(size)(image)
            return image
        self._image_transforms.append(blur_transform)

    def unsupervised_rotation(self, seed: int):
        """Switch the task to unsupervised rotation."""

        self._labels = np.random.RandomState(seed=seed).randint(4, size=self._labels.size)

        def rotate_transform(image, label):
            return torchvision.transforms.RandomRotation(label*90)(image), label
        self._joint_image_transforms.append(rotate_transform)


class ShuffleSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_examples):
        self._num_examples = num_examples
        self._seed = -1

    def __iter__(self):
        if self._seed == -1:
            indices = list(range(self._num_examples))
        elif self._seed is None:
            indices = torch.randperm(self._num_examples).tolist()
        else:
            g = torch.Generator()
            if self._seed is not None: g.manual_seed(self._seed)
            indices = torch.randperm(self._num_examples, generator=g).tolist()

        return iter(indices)

    def __len__(self):
        return self._num_examples

    def shuffle_dataorder(self, seed: int):
        self._seed = seed


class DistributedShuffleSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset):
        super(DistributedShuffleSampler, self).__init__(
            dataset, num_replicas=get_platform().world_size, rank=get_platform().rank)
        self._seed = -1

    def __iter__(self):
        indices = torch.arange(len(self.dataset))

        if self._seed != -1:
            g = torch.Generator()
            g.manual_seed(self._seed or np.random.randint(10e6))
            perm = torch.randperm(len(indices), generator=g)
            indices = indices[perm]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices.tolist())

    def shuffle_dataorder(self, seed: int):
        self._seed = seed


class DataLoader(torch.utils.data.DataLoader):
    """A wrapper that makes it possible to access the custom shuffling logic."""

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool = True):
        if get_platform().is_distributed:
            self._sampler = DistributedShuffleSampler(dataset)
        else:
            self._sampler = ShuffleSampler(len(dataset))

        self._iterations_per_epoch = np.ceil(len(dataset) / batch_size).astype(int)

        if get_platform().is_distributed:
            batch_size //= get_platform().world_size
            num_workers //= get_platform().world_size

        super(DataLoader, self).__init__(
            dataset, batch_size, sampler=self._sampler, num_workers=num_workers,
            pin_memory=pin_memory and get_platform().torch_device.type == 'cuda')

    def shuffle(self, seed: int):
        self._sampler.shuffle_dataorder(seed)

    @property
    def iterations_per_epoch(self):
        return self._iterations_per_epoch
