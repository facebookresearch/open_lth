# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from datasets import base, registry
from foundations import hparams
from testing import test_case


class TestRegistry(test_case.TestCase):
    def setUp(self):
        super(TestRegistry, self).setUp()
        self.dataset_hparams = hparams.DatasetHparams('cifar10', 50)

    def test_get(self):
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 1000)

        minibatch, labels = next(iter(loader))
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

        loader = registry.get(self.dataset_hparams, train=False)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 200)

    def test_do_not_augment(self):
        self.dataset_hparams.do_not_augment = True
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)

        minibatch, labels = next(iter(loader))
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

    def test_get_subsample(self):
        self.dataset_hparams.transformation_seed = 0
        self.dataset_hparams.subsample_fraction = 0.1
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 100)

        minibatch, labels = next(iter(loader))
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

    def test_get_random_labels(self):
        self.dataset_hparams.transformation_seed = 0
        self.dataset_hparams.random_labels_fraction = 1.0
        self.dataset_hparams.do_not_augment = True
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 1000)

        minibatch, labels = next(iter(loader))
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

    def test_get_unsupervised_labels(self):
        self.dataset_hparams.transformation_seed = 0
        self.dataset_hparams.unsupervised_labels = 'rotation'
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 1000)

        minibatch, labels = next(iter(loader))
        self.assertEqual(np.max(labels.numpy()), 3)
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

    def test_iterations_per_epoch(self):
        self.assertEqual(registry.iterations_per_epoch(self.dataset_hparams), 1000)
        self.dataset_hparams.subsample_fraction = 0.1
        self.assertEqual(registry.iterations_per_epoch(self.dataset_hparams), 100)


test_case.main()
