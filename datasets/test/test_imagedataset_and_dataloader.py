# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from datasets import base, cifar10
from testing import test_case


class TestImageDatasetAndDataLoader(test_case.TestCase):
    def test_is_image_dataset(self):
        train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        self.assertIsInstance(train_set, base.ImageDataset)

    def get_labels_from_loader(self, loader, minibatches=10):
        i = 0
        actual_labels = []
        for examples, labels in loader:
            self.assertEqual(list(examples.shape), [4, 3, 32, 32])
            self.assertEqual(list(labels.shape), [4])
            actual_labels += labels.tolist()

            i += 1
            if i > minibatches: break

        return actual_labels

    def data_loading_helper(self, dataset, shuffle_seed=None):
        loader = base.DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=True)
        self.assertIsNotNone(loader)
        self.assertEqual(len(loader), 12500)

        if shuffle_seed is not None: loader.shuffle(shuffle_seed)
        return self.get_labels_from_loader(loader)

    def test_data_loading(self):
        train_set = cifar10.Dataset.get_train_set(use_augmentation=False)
        actual_labels = self.data_loading_helper(train_set)
        self.assertEqual(actual_labels, train_set._labels[np.arange(len(actual_labels))].tolist())

        train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        self.data_loading_helper(train_set)

        train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        train_set.unsupervised_rotation(2)
        self.data_loading_helper(train_set)

        train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        train_set.blur(4)
        self.data_loading_helper(train_set)

    def test_shuffling_dataloader(self):
        train_set = cifar10.Dataset.get_train_set(use_augmentation=False)

        # The first ten labels that occur in the dataset after shuffling with seed 1.
        expected_labels = [0, 1, 6, 3, 6, 0, 0, 4, 3, 8]

        # The first ten labels seen by the data loader.
        actual_labels = self.data_loading_helper(train_set, shuffle_seed=1)[:10]

        # They should be equal.
        self.assertEqual(actual_labels, expected_labels)

        # Again for seed 0.
        expected_labels = [6, 8, 7, 5, 5, 3, 2, 2, 0, 9]
        actual_labels = self.data_loading_helper(train_set, shuffle_seed=0)[:10]
        self.assertEqual(actual_labels, expected_labels)

    def test_shuffling_dataloader_twice(self):
        train_set = cifar10.Dataset.get_train_set(use_augmentation=False)
        loader = base.DataLoader(train_set, batch_size=4, num_workers=0, pin_memory=True)

        loader.shuffle(0)
        actual_seed0 = self.get_labels_from_loader(loader)
        expected_seed0 = [6, 8, 7, 5, 5, 3, 2, 2, 0, 9]
        self.assertEqual(actual_seed0[:10], expected_seed0)

        loader.shuffle(1)
        actual_seed1 = self.get_labels_from_loader(loader)
        expected_seed1 = [0, 1, 6, 3, 6, 0, 0, 4, 3, 8]
        self.assertEqual(actual_seed1[:10], expected_seed1)

    def test_blur(self):
        train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        train_set.blur(4)

        # Nothing should change about the labels.
        self.assertEqual(len(train_set), 50000)
        self.assertEqual(np.max(train_set._labels), 9)
        self.assertEqual(train_set._labels[:10].tolist(), [6, 9, 9, 4, 1, 1, 2, 7, 8, 3])

    def test_unsupervised_rotation(self):
        train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        train_set.unsupervised_rotation(0)
        self.assertEqual(len(train_set), 50000)
        self.assertEqual(np.max(train_set._labels), 3)
        self.assertEqual(np.sum(train_set._labels == 0), 12539)
        self.assertEqual(np.sum(train_set._labels == 1), 12629)
        self.assertEqual(np.sum(train_set._labels == 2), 12411)
        self.assertEqual(np.sum(train_set._labels == 3), 12421)
        self.assertEqual(train_set._labels[:20].tolist(),
                         [0, 3, 1, 0, 3, 3, 3, 3, 1, 3, 1, 2, 0, 3, 2, 0, 0, 0, 2, 1])

        train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        train_set.unsupervised_rotation(1)
        self.assertEqual(len(train_set), 50000)
        self.assertEqual(np.max(train_set._labels), 3)
        self.assertEqual(np.sum(train_set._labels == 0), 12379)
        self.assertEqual(np.sum(train_set._labels == 1), 12568)
        self.assertEqual(np.sum(train_set._labels == 2), 12433)
        self.assertEqual(np.sum(train_set._labels == 3), 12620)
        self.assertEqual(train_set._labels[:20].tolist(),
                         [1, 3, 0, 0, 3, 1, 3, 1, 3, 0, 0, 1, 0, 3, 1, 0, 2, 1, 2, 0])


test_case.main()
