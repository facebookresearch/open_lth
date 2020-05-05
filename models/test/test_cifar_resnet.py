# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import datasets.registry
from models import cifar_resnet, initializers, registry
from testing import test_case


class TestCifarResNet(test_case.TestCase):
    def count_parameters(self, model):
        total = 0
        for _, v in model.named_parameters():
            total += np.product(list(v.shape))
        return total

    def test_valid_names(self):
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('resnet'))
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet'))
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_'))

        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_-20'))
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_0'))
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_1'))
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_6'))
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_11'))
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_18'))
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_34'))

        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_20_0'))
        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_20_-1'))

        self.assertTrue(cifar_resnet.Model.is_valid_model_name('cifar_resnet_20'))
        self.assertTrue(cifar_resnet.Model.is_valid_model_name('cifar_resnet_20_1'))
        self.assertTrue(cifar_resnet.Model.is_valid_model_name('cifar_resnet_20_16'))
        self.assertTrue(cifar_resnet.Model.is_valid_model_name('cifar_resnet_32'))
        self.assertTrue(cifar_resnet.Model.is_valid_model_name('cifar_resnet_56'))
        self.assertTrue(cifar_resnet.Model.is_valid_model_name('cifar_resnet_110'))

        self.assertFalse(cifar_resnet.Model.is_valid_model_name('cifar_resnet_20_16_16'))

    def test_resnet20(self):
        model = cifar_resnet.Model.get_model_from_name('cifar_resnet_20', initializers.kaiming_normal)
        self.assertEqual(self.count_parameters(model), 272474)

    def test_resnet110(self):
        model = cifar_resnet.Model.get_model_from_name('cifar_resnet_110', initializers.kaiming_normal)
        self.assertEqual(self.count_parameters(model), 1730714)

    def test_wrn14_128(self):
        model = cifar_resnet.Model.get_model_from_name('cifar_resnet_14_128', initializers.kaiming_normal)
        self.assertEqual(self.count_parameters(model), 11093130)

    def test_integration(self):
        default_hparams = registry.get_default_hparams('cifar_resnet_20')
        model = registry.get(default_hparams.model_hparams)
        self.assertEqual(self.count_parameters(model), 272474)

        cifar10 = datasets.registry.get(default_hparams.dataset_hparams, train=True)
        minibatch, labels = next(iter(cifar10))
        loss = model.loss_criterion(model(minibatch), labels)
        loss.backward()


test_case.main()
