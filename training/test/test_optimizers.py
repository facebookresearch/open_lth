# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from foundations import hparams
import models.registry
from training import optimizers
from testing import test_case


class TestOptimizers(test_case.TestCase):
    def setUp(self):
        super(TestOptimizers, self).setUp()
        self.hp = hparams.TrainingHparams('sgd', 0.1, '160ep')
        h = models.registry.get_default_hparams('cifar_resnet_20')
        self.model = models.registry.get(h.model_hparams)

    def test_sgd(self):
        optimizer = optimizers.get_optimizer(self.hp, self.model)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]['momentum'], 0.0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.0)

    def test_sgd_momentum(self):
        self.hp.momentum = 0.9
        optimizer = optimizers.get_optimizer(self.hp, self.model)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]['momentum'], 0.9)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.0)

    def test_sgd_weight_decay(self):
        self.hp.momentum = 0.9
        self.hp.weight_decay = 1e-4
        optimizer = optimizers.get_optimizer(self.hp, self.model)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]['momentum'], 0.9)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-4)

    def test_adam(self):
        self.hp.optimizer_name = 'adam'
        optimizer = optimizers.get_optimizer(self.hp, self.model)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.0)

    def test_adam_weight_decay(self):
        self.hp.optimizer_name = 'adam'
        self.hp.weight_decay = 1e-4
        optimizer = optimizers.get_optimizer(self.hp, self.model)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-4)

    def test_nonexistent_optimizer(self):
        self.hp.optimizer_name = 'metagrad'
        with self.assertRaises(ValueError):
            optimizers.get_optimizer(self.hp, self.model)


test_case.main()
