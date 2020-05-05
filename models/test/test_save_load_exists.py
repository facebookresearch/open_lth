# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

from foundations import paths
from foundations.step import Step
from models import registry
from testing import test_case


class TestSaveLoadExists(test_case.TestCase):
    def test_save_load_exists(self):
        hp = registry.get_default_hparams('cifar_resnet_20')
        model = registry.get(hp.model_hparams)
        step = Step.from_iteration(27, 17)
        model_location = paths.model(self.root, step)
        model_state = TestSaveLoadExists.get_state(model)

        self.assertFalse(registry.exists(self.root, step))
        self.assertFalse(os.path.exists(model_location))

        # Test saving.
        model.save(self.root, step)
        self.assertTrue(registry.exists(self.root, step))
        self.assertTrue(os.path.exists(model_location))

        # Test loading.
        model = registry.get(hp.model_hparams)
        model.load_state_dict(torch.load(model_location))
        self.assertStateEqual(model_state, TestSaveLoadExists.get_state(model))

        model = registry.load(self.root, step, hp.model_hparams)
        self.assertStateEqual(model_state, TestSaveLoadExists.get_state(model))


test_case.main()
