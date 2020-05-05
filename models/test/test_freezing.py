# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models import registry
from testing import test_case


class TestFreezing(test_case.TestCase):
    def test_freeze_batchnorm(self):
        default_hparams = registry.get_default_hparams('cifar_resnet_20')
        default_hparams.model_hparams.batchnorm_frozen = True
        model = registry.get(default_hparams.model_hparams)

        bn_names = []
        for k, v in model.named_modules():
            if isinstance(v, torch.nn.BatchNorm2d):
                bn_names += [k + '.weight', k + '.bias']

        for k, v in model.named_parameters():
            with self.subTest(tensor=k):
                if k in bn_names:
                    self.assertFalse(v.requires_grad)
                else:
                    self.assertTrue(v.requires_grad)

    def test_freeze_output(self):
        default_hparams = registry.get_default_hparams('cifar_resnet_20')
        default_hparams.model_hparams.output_frozen = True
        model = registry.get(default_hparams.model_hparams)

        for k, v in model.named_parameters():
            with self.subTest(tensor=k):
                if k in model.output_layer_names:
                    self.assertFalse(v.requires_grad)
                else:
                    self.assertTrue(v.requires_grad)

    def test_freeze_others(self):
        default_hparams = registry.get_default_hparams('cifar_resnet_20')
        default_hparams.model_hparams.others_frozen = True
        model = registry.get(default_hparams.model_hparams)

        enabled_names = []
        for k, v in model.named_modules():
            if isinstance(v, torch.nn.BatchNorm2d):
                enabled_names += [k + '.weight', k + '.bias']
        enabled_names += model.output_layer_names

        for k, v in model.named_parameters():
            with self.subTest(tensor=k):
                if k in enabled_names:
                    self.assertTrue(v.requires_grad)
                else:
                    self.assertFalse(v.requires_grad)

    def test_freeze_all(self):
        default_hparams = registry.get_default_hparams('cifar_resnet_20')
        default_hparams.model_hparams.others_frozen = True
        default_hparams.model_hparams.output_frozen = True
        default_hparams.model_hparams.batchnorm_frozen = True
        model = registry.get(default_hparams.model_hparams)

        for k, v in model.named_parameters():
            with self.subTest(tensor=k):
                self.assertFalse(v.requires_grad)

    def test_freeze_none(self):
        default_hparams = registry.get_default_hparams('cifar_resnet_20')
        model = registry.get(default_hparams.model_hparams)

        for k, v in model.named_parameters():
            with self.subTest(tensor=k):
                self.assertTrue(v.requires_grad)


test_case.main()
