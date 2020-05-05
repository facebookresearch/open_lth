# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import numpy as np
import os
import torch

import datasets.registry
from foundations import paths
from foundations.step import Step
from lottery.runner import LotteryRunner
import models.registry
from pruning.mask import Mask
from testing import test_case


class TestRunner(test_case.TestCase):
    def setUp(self):
        super(TestRunner, self).setUp()
        self.desc = models.registry.get_default_hparams('cifar_resnet_8_2')

    def to_step(self, s):
        return Step.from_str(s, datasets.registry.iterations_per_epoch(self.desc.dataset_hparams))

    def assertLevelFilesPresent(self, level_root, start_step, end_step, masks=False):
        with self.subTest(level_root=level_root):
            self.assertTrue(os.path.exists(paths.model(level_root, start_step)))
            self.assertTrue(os.path.exists(paths.model(level_root, end_step)))
            self.assertTrue(os.path.exists(paths.logger(level_root)))
            if masks:
                self.assertTrue(os.path.exists(paths.mask(level_root)))
                self.assertTrue(os.path.exists(paths.sparsity_report(level_root)))

    def test_level0_2it(self):
        self.desc.training_hparams.training_steps = '2it'
        LotteryRunner(replicate=2, levels=0, desc=self.desc, verbose=False).run()
        level_root = self.desc.run_path(2, 0)

        # Ensure the important files are there.
        self.assertLevelFilesPresent(level_root, self.to_step('0it'), self.to_step('2it'))

        # Ensure that the mask is all 1's.
        mask = Mask.load(level_root)
        for v in mask.numpy().values(): self.assertTrue(np.all(np.equal(v, 1)))
        with open(paths.sparsity_report(level_root)) as fp:
            sparsity_report = json.loads(fp.read())
        self.assertEqual(sparsity_report['unpruned'] / sparsity_report['total'], 1)

    def test_level3_2it(self):
        self.desc.training_hparams.training_steps = '2it'
        LotteryRunner(replicate=2, levels=3, desc=self.desc, verbose=False).run()

        level0_weights = paths.model(self.desc.run_path(2, 0), self.to_step('0it'))
        level0_weights = {k: v.numpy() for k, v in torch.load(level0_weights).items()}

        for level in range(0, 4):
            level_root = self.desc.run_path(2, level)
            self.assertLevelFilesPresent(level_root, self.to_step('0it'), self.to_step('2it'))

            # Check the mask.
            pct = 0.8**level
            mask = Mask.load(level_root).numpy()

            # Check the mask itself.
            total, total_present = 0.0, 0.0
            for v in mask.values():
                total += v.size
                total_present += np.sum(v)
            self.assertTrue(np.allclose(pct, total_present / total, atol=0.01))

            # Check the sparsity report.
            with open(paths.sparsity_report(level_root)) as fp:
                sparsity_report = json.loads(fp.read())
            self.assertTrue(np.allclose(pct, sparsity_report['unpruned'] / sparsity_report['total'], atol=0.01))

            # Ensure that the initial weights are a masked version of the level 0 weights.
            level_weights = paths.model(level_root, self.to_step('0it'))
            level_weights = {k: v.numpy() for k, v in torch.load(level_weights).items()}
            self.assertStateEqual(level_weights, {k: v * mask.get(k, 1) for k, v in level0_weights.items()})

    def test_level3_4it_pretrain2it(self):
        self.desc.pretrain_dataset_hparams = copy.deepcopy(self.desc.dataset_hparams)
        self.desc.pretrain_training_hparams = copy.deepcopy(self.desc.training_hparams)
        self.desc.pretrain_training_hparams.training_steps = '2it'
        self.desc.training_hparams.training_steps = '4it'
        LotteryRunner(replicate=2, levels=3, desc=self.desc, verbose=False).run()

        # Check that the pretrain weights are present.
        pretrain_root = self.desc.run_path(2, 'pretrain')
        self.assertLevelFilesPresent(pretrain_root, self.to_step('0it'), self.to_step('2it'), masks=False)

        # Load the pretrain and level0 start weights to ensure they're the same.
        pretrain_end_weights = paths.model(self.desc.run_path(2, 'pretrain'), self.desc.pretrain_end_step)
        pretrain_end_weights = {k: v.numpy() for k, v in torch.load(pretrain_end_weights).items()}

        level0_weights = paths.model(self.desc.run_path(2, 0), self.desc.train_start_step)
        level0_weights = {k: v.numpy() for k, v in torch.load(level0_weights).items()}

        self.assertStateEqual(pretrain_end_weights, level0_weights)

        # Evaluate each of the pruning levels.
        for level in range(0, 2):
            level_root = self.desc.run_path(2, level)
            self.assertLevelFilesPresent(level_root, self.to_step('2it'), self.to_step('4it'))

            # Ensure that the initial weights are a masked version of the level 0 weights
            # (which are identical to the weights at the end of pretraining).
            mask = Mask.load(level_root).numpy()
            level_weights = paths.model(level_root, self.desc.train_start_step)
            level_weights = {k: v.numpy() for k, v in torch.load(level_weights).items()}
            self.assertStateEqual(level_weights, {k: v * mask.get(k, 1) for k, v in level0_weights.items()})

    def test_level3_4it_pretrain2it_different_output_size(self):
        self.desc.pretrain_dataset_hparams = copy.deepcopy(self.desc.dataset_hparams)
        self.desc.pretrain_training_hparams = copy.deepcopy(self.desc.training_hparams)
        self.desc.pretrain_training_hparams.training_steps = '2it'
        self.desc.pretrain_dataset_hparams.unsupervised_labels = 'rotation'
        self.desc.training_hparams.training_steps = '4it'
        LotteryRunner(replicate=2, levels=3, desc=self.desc, verbose=False).run()

        # Check that the pretrain weights are present.
        pretrain_root = self.desc.run_path(2, 'pretrain')
        self.assertLevelFilesPresent(pretrain_root, self.to_step('0it'), self.to_step('2it'), masks=False)

        # Load the pretrain and level0 start weights to ensure they're the same.
        pretrain_end_weights = paths.model(self.desc.run_path(2, 'pretrain'), self.desc.pretrain_end_step)
        pretrain_end_weights = {k: v.numpy() for k, v in torch.load(pretrain_end_weights).items()}

        level0_weights = paths.model(self.desc.run_path(2, 0), self.desc.train_start_step)
        level0_weights = {k: v.numpy() for k, v in torch.load(level0_weights).items()}

        # All weights should be identical except for the output layers.
        output_layer_names = models.registry.get(self.desc.model_hparams).output_layer_names
        self.assertStateEqual({k: v for k, v in pretrain_end_weights.items() if k not in output_layer_names},
                              {k: v for k, v in level0_weights.items() if k not in output_layer_names})

        # Evaluate each of the pruning levels.
        for level in range(0, 2):
            level_root = self.desc.run_path(2, level)
            self.assertLevelFilesPresent(level_root, self.to_step('2it'), self.to_step('4it'))

            # Ensure that the initial weights are a masked version of the level 0 weights
            # (which are identical to the weights at the end of pretraining).
            mask = Mask.load(level_root).numpy()
            level_weights = paths.model(level_root, self.desc.train_start_step)
            level_weights = {k: v.numpy() for k, v in torch.load(level_weights).items()}
            self.assertStateEqual(level_weights, {k: v * mask.get(k, 1) for k, v in level0_weights.items()})


test_case.main()
