# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import unittest
import shutil

import platforms.local
import platforms.platform


class Platform(platforms.local.Platform):
    @property
    def device_str(self):
        return 'cpu'

    @property
    def is_parallel(self):
        return False

    @property
    def root(self):
        return os.path.join(super(Platform, self).root, 'TESTING')


class TestCase(unittest.TestCase):
    def setUp(self):
        self.saved_platform = platforms.platform._PLATFORM
        platforms.platform._PLATFORM = Platform(num_workers=4)
        if os.path.exists(platforms.platform.get_platform().root):
            shutil.rmtree(platforms.platform.get_platform().root)
        os.makedirs(platforms.platform.get_platform().root)
        self.root = platforms.platform.get_platform().root

    def tearDown(self):
        if os.path.exists(self.root): shutil.rmtree(self.root)
        platforms.platform._PLATFORM = self.saved_platform

    @staticmethod
    def get_state(model):
        """Get a copy of the state of a model."""

        return {k: v.clone().detach().cpu().numpy() for k, v in model.state_dict().items()}

    def assertStateEqual(self, state1, state2):
        """Assert that two models states are equal."""

        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for k in state1:
            self.assertTrue(np.array_equal(state1[k], state2[k]))

    def assertStateAllNotEqual(self, state1, state2):
        """Assert that two models states are not equal in any tensor."""

        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for k in state1:
            self.assertFalse(np.array_equal(state1[k], state2[k]))


def main():
    if __name__ == '__main__':
        unittest.main()
