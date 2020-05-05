# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch

from foundations import paths
import models.registry
from pruning.mask import Mask
from testing import test_case


class TestMask(test_case.TestCase):
    def test_dict_behavior(self):
        m = Mask()
        self.assertEqual(len(m), 0)
        self.assertEqual(len(m.keys()), 0)
        self.assertEqual(len(m.values()), 0)

        m['hello'] = np.ones([2, 3])
        m['world'] = np.zeros([5, 6])
        self.assertEqual(len(m), 2)
        self.assertEqual(len(m.keys()), 2)
        self.assertEqual(len(m.values()), 2)
        self.assertEqual(set(m.keys()), set(['hello', 'world']))
        self.assertTrue(np.array_equal(np.ones([2, 3]), m['hello']))
        self.assertTrue(np.array_equal(np.zeros([5, 6]), m['world']))

        del m['hello']
        self.assertEqual(len(m), 1)
        self.assertEqual(len(m.keys()), 1)
        self.assertEqual(len(m.values()), 1)
        self.assertEqual(set(m.keys()), set(['world']))
        self.assertTrue(np.array_equal(np.zeros([5, 6]), m['world']))

    def test_create_mask_from_dict(self):
        m = Mask({'hello': np.ones([2, 3]), 'world': np.zeros([5, 6])})
        self.assertEqual(len(m), 2)
        self.assertEqual(len(m.keys()), 2)
        self.assertEqual(len(m.values()), 2)
        self.assertEqual(set(m.keys()), set(['hello', 'world']))
        self.assertTrue(np.array_equal(np.ones([2, 3]), m['hello']))
        self.assertTrue(np.array_equal(np.zeros([5, 6]), m['world']))

    def test_create_from_tensor(self):
        m = Mask({'hello': torch.ones([2, 3]), 'world': torch.zeros([5, 6])})
        self.assertEqual(len(m), 2)
        self.assertEqual(len(m.keys()), 2)
        self.assertEqual(len(m.values()), 2)
        self.assertEqual(set(m.keys()), set(['hello', 'world']))
        self.assertTrue(np.array_equal(np.ones([2, 3]), m['hello']))
        self.assertTrue(np.array_equal(np.zeros([5, 6]), m['world']))

    def test_bad_inputs(self):
        m = Mask()

        with self.assertRaises(ValueError):
            m[''] = np.ones([2, 3])

        with self.assertRaises(ValueError):
            m[6] = np.ones([2, 3])

        with self.assertRaises(ValueError):
            m['hello'] = [[0, 1], [1, 0]]

        with self.assertRaises(ValueError):
            m['hello'] = np.array([[0, 1], [2, 0]])

    def test_ones_like(self):
        model = models.registry.get(models.registry.get_default_hparams('cifar_resnet_20').model_hparams)
        m = Mask.ones_like(model)

        for k, v in model.state_dict().items():
            if k in model.prunable_layer_names:
                self.assertIn(k, m)
                self.assertEqual(list(m[k].shape), list(v.shape))
                self.assertTrue((m[k] == 1).all())
            else:
                self.assertNotIn(k, m)

    def test_save_load_exists(self):
        self.assertFalse(Mask.exists(self.root))
        self.assertFalse(os.path.exists(paths.mask(self.root)))

        m = Mask({'hello': np.ones([2, 3]), 'world': np.zeros([5, 6])})
        m.save(self.root)
        self.assertTrue(os.path.exists(paths.mask(self.root)))
        self.assertTrue(Mask.exists(self.root))

        m2 = Mask.load(self.root)
        self.assertEqual(len(m2), 2)
        self.assertEqual(len(m2.keys()), 2)
        self.assertEqual(len(m2.values()), 2)
        self.assertEqual(set(m2.keys()), set(['hello', 'world']))
        self.assertTrue(np.array_equal(np.ones([2, 3]), m2['hello']))
        self.assertTrue(np.array_equal(np.zeros([5, 6]), m2['world']))


test_case.main()
