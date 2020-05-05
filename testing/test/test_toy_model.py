# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from testing import test_case
from testing.toy_model import InnerProductModel


class TestInnerProductModel(test_case.TestCase):
    def setUp(self):
        super(TestInnerProductModel, self).setUp()
        self.model = InnerProductModel(10)
        self.example = torch.ones(1, 10)
        self.label = torch.ones(1) * 40.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def test_forward(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(45.0, self.model(self.example).item())

    def test_backward(self):
        self.model.train()

        self.optimizer.zero_grad()
        self.model.loss_criterion(self.model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertEqual(44.0, self.model(self.example).item())

        self.optimizer.zero_grad()
        self.model.loss_criterion(self.model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertEqual(43.2, np.round(self.model(self.example).item(), 4))

        self.optimizer.zero_grad()
        self.model.loss_criterion(self.model(self.example), self.label).backward()
        self.optimizer.step()
        self.assertEqual(42.56, np.round(self.model(self.example).item(), 4))


test_case.main()
