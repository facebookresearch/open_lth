# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import models.registry
from pruning.sparse_global import Strategy
from pruning.sparse_global import PruningHparams
from testing import test_case


class TestSparseGlobal(test_case.TestCase):
    def setUp(self):
        super(TestSparseGlobal, self).setUp()
        self.hparams = PruningHparams('sparse_global', 0.2)

        model_hparams = models.registry.get_default_hparams('cifar_resnet_20').model_hparams
        self.model = models.registry.get(model_hparams)

    def test_get_pruning_hparams(self):
        self.assertTrue(issubclass(Strategy.get_pruning_hparams(), PruningHparams))

    def test_prune(self):
        m = Strategy.prune(self.hparams, self.model)

        # Check that the mask only contains entries for the prunable layers.
        self.assertEqual(set(m.keys()), set(self.model.prunable_layer_names))

        # Check that the masks are the same sizes as the tensors.
        for k in self.model.prunable_layer_names:
            self.assertEqual(list(m[k].shape), list(self.model.state_dict()[k].shape))

        # Check that the right fraction of weights was pruned among prunable layers.
        m = m.numpy()
        total_pruned = np.sum([np.sum(1 - v) for v in m.values()])
        total_weights = np.sum([v.size for v in m.values()])
        actual_fraction = float(total_pruned) / total_weights
        self.assertGreaterEqual(actual_fraction, self.hparams.pruning_fraction)
        self.assertGreater(self.hparams.pruning_fraction + 0.0001, actual_fraction)

        # Ensure that the right threshold was chosen.
        pruned_weights = [self.model.state_dict()[k].numpy()[m[k] == 0] for k in m]
        threshold = np.max(np.abs(np.concatenate(pruned_weights)))
        unpruned_weights = [self.model.state_dict()[k].numpy()[m[k] == 1] for k in m]
        self.assertTrue(np.all(np.abs(np.concatenate(unpruned_weights)) > threshold))

    def test_iterative_pruning(self):
        m = Strategy.prune(self.hparams, self.model)
        m2 = Strategy.prune(self.hparams, self.model, m)

        # Ensure that all weights pruned before are still pruned here.
        m, m2 = m.numpy(), m2.numpy()
        self.assertEqual(set(m.keys()), set(m2.keys()))
        for k in m:
            self.assertTrue(np.all(m[k] >= m2[k]))

        total_pruned = np.sum([np.sum(1 - v) for v in m2.values()])
        total_weights = np.sum([v.size for v in m2.values()])
        actual_fraction = float(total_pruned) / total_weights
        expected_fraction = 1 - (1 - self.hparams.pruning_fraction) ** 2
        self.assertGreaterEqual(actual_fraction, expected_fraction)
        self.assertGreater(expected_fraction + 0.0001, actual_fraction)

    def test_prune_layers_to_ignore(self):
        layers_to_ignore = sorted(self.model.prunable_layer_names)[:5]
        self.hparams.pruning_layers_to_ignore = ','.join(layers_to_ignore)

        m = Strategy.prune(self.hparams, self.model).numpy()

        # Ensure that the ignored layers were, indeed, ignored.
        for k in layers_to_ignore:
            self.assertTrue(np.all(m[k] == 1))

        # Ensure that the expected fraction of parameters was still pruned.
        total_pruned = np.sum([np.sum(1 - v) for v in m.values()])
        total_weights = np.sum([v.size for v in m.values()])
        actual_fraction = float(total_pruned) / total_weights
        self.assertGreaterEqual(actual_fraction, self.hparams.pruning_fraction)
        self.assertGreater(self.hparams.pruning_fraction + 0.0001, actual_fraction)


test_case.main()
