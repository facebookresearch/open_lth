# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from testing import test_case
from utils import tensor_utils


class TestTensorUtils(test_case.TestCase):
    def test_vectorize(self):
        state_dict = {
            'x': torch.arange(5),
            'z': torch.arange(100, 105),
            'y': torch.eye(3).long()
        }
        v = tensor_utils.vectorize(state_dict)
        self.assertEqual(v.tolist(), [0, 1, 2, 3, 4, 1, 0, 0, 0, 1, 0, 0, 0, 1, 100, 101, 102, 103, 104])

    def test_unvectorize(self):
        reference_state_dict = {
            'x': torch.ones(5),
            'z': torch.ones(5),
            'y': torch.ones([3, 3]),
        }
        v = torch.tensor([0, 1, 2, 3, 4, 1, 0, 0, 0, 1, 0, 0, 0, 1, 100, 101, 102, 103, 104])
        state_dict = tensor_utils.unvectorize(v, reference_state_dict)

        self.assertEqual(set(reference_state_dict.keys()), set(state_dict.keys()))
        for k in state_dict:
            self.assertEqual(list(state_dict[k].shape), list(reference_state_dict[k].shape))

        self.assertEqual(state_dict['x'].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(state_dict['z'].tolist(), [100, 101, 102, 103, 104])
        self.assertEqual(state_dict['y'].tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_perm(self):
        p = tensor_utils.perm(10)
        self.assertEqual([10], list(p.shape))
        self.assertEqual(set(range(10)), set(p.tolist()))

        p = tensor_utils.perm(10, 0)
        self.assertEqual([10], list(p.shape))
        self.assertEqual(set(range(10)), set(p.tolist()))
        self.assertEqual(p.tolist(), [2, 5, 4, 8, 9, 1, 6, 3, 7, 0])

        p = tensor_utils.perm(10, 0)
        self.assertEqual(p.tolist(), [2, 5, 4, 8, 9, 1, 6, 3, 7, 0])

        p = tensor_utils.perm(10, 1)
        self.assertEqual(p.tolist(), [6, 8, 9, 4, 5, 2, 1, 7, 3, 0])

    def test_shuffle_tensor(self):
        t = torch.eye(5)
        s = tensor_utils.shuffle_tensor(t)
        self.assertEqual(list(t.shape), list(s.shape))
        self.assertEqual(sorted(t.view(-1).tolist()), sorted(s.view(-1).tolist()))
        self.assertEqual(torch.sum(t).item(), torch.sum(s).item())

        t = torch.tensor(range(10, 20))
        s = tensor_utils.shuffle_tensor(t)
        self.assertEqual(set(t.tolist()), set(s.tolist()))

        s = tensor_utils.shuffle_tensor(t, 0)
        self.assertEqual(s.tolist(), [12, 15, 14, 18, 19, 11, 16, 13, 17, 10])

        s = tensor_utils.shuffle_tensor(t, 0)
        self.assertEqual(s.tolist(), [12, 15, 14, 18, 19, 11, 16, 13, 17, 10])

        s = tensor_utils.shuffle_tensor(t, 1)
        self.assertEqual(s.tolist(), [16, 18, 19, 14, 15, 12, 11, 17, 13, 10])


test_case.main()
