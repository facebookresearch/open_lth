# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from datasets import mnist
from testing import test_case


class TestDataset(test_case.TestCase):
    def setUp(self):
        super(TestDataset, self).setUp()
        self.test_set = mnist.Dataset.get_test_set()
        self.train_set = mnist.Dataset.get_train_set(use_augmentation=True)

    def test_size(self):
        self.assertEqual(mnist.Dataset.num_classes(), 10)
        self.assertEqual(mnist.Dataset.num_train_examples(), 60000)
        self.assertEqual(mnist.Dataset.num_test_examples(), 10000)

    def test_not_none(self):
        self.assertIsNotNone(self.test_set)
        self.assertIsNotNone(self.train_set)

    def test_randomize_labels_half(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0.5)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 5472)

    def test_randomize_labels_none(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 0)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 10000)

    def test_randomize_labels_all(self):
        labels_before = self.test_set._labels.tolist()
        self.test_set.randomize_labels(0, 1)
        examples_match = np.sum(np.equal(labels_before, self.test_set._labels).astype(int))
        self.assertEqual(examples_match, 989)

    def test_subsample(self):
        # Subsample the test set.
        labels_test = [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
        subsampled_labels_with_seed_zero_test = [6, 9, 2, 6, 7, 6, 1, 4, 7, 1]

        self.assertEqual(self.test_set._labels[:10].tolist(), labels_test)
        self.test_set.subsample(0, 0.1)
        self.assertEqual(len(self.test_set), 1000)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsampled_labels_with_seed_zero_test)

        # Evaluate with a different seed.
        subsampled_labels_with_seed_one_test = [8, 8, 7, 0, 9, 8, 0, 2, 3, 0]
        self.test_set = mnist.Dataset.get_test_set()
        self.test_set.subsample(1, 0.1)
        self.assertEqual(len(self.test_set), 1000)
        self.assertEqual(self.test_set._labels[:10].tolist(), subsampled_labels_with_seed_one_test)

        # Subsample the train set.
        labels_train = [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]
        subsampled_labels_with_seed_zero_train = [3, 2, 7, 8, 2, 3, 5, 2, 8, 9]

        self.assertEqual(self.train_set._labels[:10].tolist(), labels_train)
        self.train_set.subsample(0, 0.1)
        self.assertEqual(len(self.train_set), 6000)
        self.assertEqual(self.train_set._labels[:10].tolist(), subsampled_labels_with_seed_zero_train)

        # Evaluate with a different seed.
        subsampled_labels_with_seed_one_train = [2, 1, 5, 0, 6, 0, 8, 9, 5, 5]
        self.train_set = mnist.Dataset.get_train_set(use_augmentation=True)
        self.train_set.subsample(1, 0.1)
        self.assertEqual(len(self.train_set), 6000)
        self.assertEqual(self.train_set._labels[:10].tolist(), subsampled_labels_with_seed_one_train)

    def test_subsample_twice(self):
        self.train_set.subsample(1, 0.1)
        with self.assertRaises(ValueError):
            self.train_set.subsample(1, 0.1)


test_case.main()
