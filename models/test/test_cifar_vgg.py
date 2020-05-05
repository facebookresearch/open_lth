# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import datasets.registry
from models import cifar_vgg, initializers, registry
from testing import test_case


class TestCifarVGG(test_case.TestCase):
    def count_parameters(self, model):
        total = 0
        for _, v in model.named_parameters():
            total += np.product(list(v.shape))
        return total

    def test_valid_names(self):
        self.assertFalse(cifar_vgg.Model.is_valid_model_name('vgg'))
        self.assertFalse(cifar_vgg.Model.is_valid_model_name('cifar_vgg'))
        self.assertFalse(cifar_vgg.Model.is_valid_model_name('cifar_vgg_'))

        self.assertFalse(cifar_vgg.Model.is_valid_model_name('cifar_vgg_-11'))
        self.assertFalse(cifar_vgg.Model.is_valid_model_name('cifar_vgg_0'))

        self.assertTrue(cifar_vgg.Model.is_valid_model_name('cifar_vgg_11'))
        self.assertTrue(cifar_vgg.Model.is_valid_model_name('cifar_vgg_13'))
        self.assertTrue(cifar_vgg.Model.is_valid_model_name('cifar_vgg_16'))
        self.assertTrue(cifar_vgg.Model.is_valid_model_name('cifar_vgg_19'))

        self.assertFalse(cifar_vgg.Model.is_valid_model_name('cifar_vgg_16_2'))

    def test_vgg11(self):
        model = cifar_vgg.Model.get_model_from_name('cifar_vgg_11', initializers.kaiming_normal)
        self.assertEqual(self.count_parameters(model), 9231114)

    def test_vgg13(self):
        model = cifar_vgg.Model.get_model_from_name('cifar_vgg_13', initializers.kaiming_normal)
        self.assertEqual(self.count_parameters(model), 9416010)

    def test_vgg16(self):
        model = cifar_vgg.Model.get_model_from_name('cifar_vgg_16', initializers.kaiming_normal)
        self.assertEqual(self.count_parameters(model), 14728266)

    def test_vgg19(self):
        model = cifar_vgg.Model.get_model_from_name('cifar_vgg_19', initializers.kaiming_normal)
        self.assertEqual(self.count_parameters(model), 20040522)

    def test_integration(self):
        default_hparams = registry.get_default_hparams('cifar_vgg_13')
        model = registry.get(default_hparams.model_hparams, outputs=10)
        self.assertEqual(self.count_parameters(model), 9416010)

        cifar10 = datasets.registry.get(default_hparams.dataset_hparams, train=True)
        minibatch, labels = next(iter(cifar10))
        loss = model.loss_criterion(model(minibatch), labels)
        loss.backward()


test_case.main()
