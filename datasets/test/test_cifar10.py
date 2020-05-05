# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from datasets import cifar10
from testing import test_case


class TestDataset(test_case.TestCase):
    def setUp(self):
        super(TestDataset, self).setUp()
        self.test_set = cifar10.Dataset.get_test_set()
        self.train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        self.train_set_noaugment = cifar10.Dataset.get_train_set(False)

    def test_not_none(self):
        self.assertIsNotNone(self.test_set)
        self.assertIsNotNone(self.train_set)
        self.assertIsNotNone(self.train_set_noaugment)

    def test_size(self):
        self.assertEqual(cifar10.Dataset.num_classes(), 10)
        self.assertEqual(cifar10.Dataset.num_train_examples(), 50000)
        self.assertEqual(cifar10.Dataset.num_test_examples(), 10000)

    def test_randomize_labels_half(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0.5)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 5503)

    def test_randomize_labels_none(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 10000)

    def test_randomize_labels_all(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 1)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 1020)

    def test_subsample(self):
        # Subsample the test set.
        labels_test = [3, 8, 8, 0, 6, 6, 1, 6, 3, 1]
        subsampled_labels_with_seed_zero_test = [7, 9, 3, 8, 0, 1, 0, 6, 3, 7]

        self.assertEqual(self.test_set._labels[:10].tolist(), labels_test)
        self.test_set.subsample(0, 0.1)
        self.assertEqual(len(self.test_set), 1000)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsampled_labels_with_seed_zero_test)

        # Evaluate with a different seed.
        subsampled_labels_with_seed_one_test = [1, 6, 4, 7, 9, 1, 7, 2, 8, 5]
        self.test_set = cifar10.Dataset.get_test_set()
        self.test_set.subsample(1, 0.1)
        self.assertEqual(len(self.test_set), 1000)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsampled_labels_with_seed_one_test)

        # Subsample the train set.
        labels_train = [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]
        subsampled_labels_with_seed_zero_train = [6, 7, 3, 6, 8, 1, 2, 9, 5, 2]

        self.assertEqual(self.train_set._labels[:10].tolist(), labels_train)
        self.train_set.subsample(0, 0.1)
        self.assertEqual(len(self.train_set), 5000)
        self.assertEqual(self.train_set._labels[:10].tolist(), subsampled_labels_with_seed_zero_train)

        # Evaluate with a different seed.
        subsampled_labels_with_seed_one_train = [2, 1, 4, 2, 5, 6, 4, 3, 8, 2]
        self.train_set = cifar10.Dataset.get_train_set(use_augmentation=True)
        self.train_set.subsample(1, 0.1)
        self.assertEqual(len(self.train_set), 5000)
        self.assertEqual(self.train_set._labels[:10].tolist(), subsampled_labels_with_seed_one_train)

    def test_subsample_twice(self):
        self.train_set.subsample(1, 0.1)
        with self.assertRaises(ValueError):
            self.train_set.subsample(1, 0.1)


test_case.main()
