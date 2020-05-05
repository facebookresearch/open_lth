# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import warnings

from foundations import hparams
import models.registry
from training import optimizers
from testing import test_case


class TestLrScheduler(test_case.TestCase):
    def setUp(self):
        super(TestLrScheduler, self).setUp()
        self.hp = hparams.TrainingHparams(
            optimizer_name='sgd',
            lr=0.1,
            training_steps='160ep'
        )

        h = models.registry.get_default_hparams('cifar_resnet_20')
        self.model = models.registry.get(h.model_hparams)
        self.optimizer = optimizers.get_optimizer(self.hp, self.model)

    def assertLrEquals(self, lr):
        self.assertEqual(np.round(self.optimizer.param_groups[0]['lr'], 10), np.round(lr, 10))

    def test_vanilla(self):
        with warnings.catch_warnings():  # Filter unnecessary warning.
            warnings.filterwarnings("ignore", category=UserWarning)
            lrs = optimizers.get_lr_schedule(self.hp, self.optimizer, 10)
            self.assertLrEquals(0.1)
            for _ in range(100): lrs.step()
            self.assertLrEquals(0.1)

    def test_milestones(self):
        with warnings.catch_warnings():  # Filter unnecessary warning.
            warnings.filterwarnings("ignore", category=UserWarning)
            self.hp.gamma = 0.1
            self.hp.milestone_steps = '2ep,4ep,7ep,8ep'

            lrs = optimizers.get_lr_schedule(self.hp, self.optimizer, 10)

            self.assertLrEquals(0.1)
            for _ in range(19): lrs.step()
            self.assertLrEquals(1e-1)

            for _ in range(1): lrs.step()
            self.assertLrEquals(1e-2)
            for _ in range(19): lrs.step()
            self.assertLrEquals(1e-2)

            for _ in range(1): lrs.step()
            self.assertLrEquals(1e-3)
            for _ in range(29): lrs.step()
            self.assertLrEquals(1e-3)

            for _ in range(1): lrs.step()
            self.assertLrEquals(1e-4)
            for _ in range(9): lrs.step()
            self.assertLrEquals(1e-4)

            for _ in range(1): lrs.step()
            self.assertLrEquals(1e-5)
            for _ in range(100): lrs.step()
            self.assertLrEquals(1e-5)

    def test_warmup(self):
        with warnings.catch_warnings():  # Filter unnecessary warning.
            warnings.filterwarnings("ignore", category=UserWarning)
            self.hp.warmup_steps = '20it'
            lrs = optimizers.get_lr_schedule(self.hp, self.optimizer, 10)

            for i in range(20):
                self.assertLrEquals(i / 20 * 0.1)
                lrs.step()
            self.assertLrEquals(0.1)
            for i in range(100):
                lrs.step()
                self.assertLrEquals(0.1)


test_case.main()
