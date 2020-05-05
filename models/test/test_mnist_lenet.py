# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import datasets.registry
from models import mnist_lenet, initializers, registry
from testing import test_case


class TestMnistLenet(test_case.TestCase):
    def count_parameters(self, model):
        total = 0
        for _, v in model.named_parameters():
            total += np.product(list(v.shape))
        return total

    def test_valid_names(self):
        self.assertFalse(mnist_lenet.Model.is_valid_model_name('lenet'))
        self.assertFalse(mnist_lenet.Model.is_valid_model_name('mnist_lenet'))
        self.assertFalse(mnist_lenet.Model.is_valid_model_name('mnist_lenet_'))
        self.assertFalse(mnist_lenet.Model.is_valid_model_name('mnist_lenet_0'))
        self.assertFalse(mnist_lenet.Model.is_valid_model_name('mnist_lenet_-1'))
        self.assertFalse(mnist_lenet.Model.is_valid_model_name('mnist_lenet_lenet'))

        self.assertTrue(mnist_lenet.Model.is_valid_model_name('mnist_lenet_300'))
        self.assertTrue(mnist_lenet.Model.is_valid_model_name('mnist_lenet_300_100'))
        self.assertTrue(mnist_lenet.Model.is_valid_model_name('mnist_lenet_300_100_5000'))

    def test_lenet300_100(self):
        model = mnist_lenet.Model.get_model_from_name('mnist_lenet_300_100', initializers.kaiming_normal)
        self.assertEqual(self.count_parameters(model), 266610)

    def test_lenet500_500_500(self):
        model = mnist_lenet.Model.get_model_from_name('mnist_lenet_500_500_500', initializers.kaiming_normal)
        self.assertEqual(self.count_parameters(model), 898510)

    def test_integration(self):
        default_hparams = registry.get_default_hparams('mnist_lenet_300_100')
        model = registry.get(default_hparams.model_hparams)
        self.assertEqual(self.count_parameters(model), 266610)

        mnist = datasets.registry.get(default_hparams.dataset_hparams, train=True)
        minibatch, labels = next(iter(mnist))
        loss = model.loss_criterion(model(minibatch), labels)
        loss.backward()


test_case.main()
