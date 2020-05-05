# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from foundations.step import Step
from testing import test_case


class TestStep(test_case.TestCase):
    def assertStepEquals(self, step, iteration, ep, it):
        self.assertEqual(step.iteration, iteration)
        self.assertEqual(step.ep, ep)
        self.assertEqual(step.it, it)

    def test_constructor(self):
        with self.assertRaises(ValueError):
            Step(-1, 20)

        with self.assertRaises(ValueError):
            Step(1, 0)

        with self.assertRaises(ValueError):
            Step(1, -5)

        self.assertStepEquals(Step(0, 1), 0, 0, 0)
        self.assertStepEquals(Step(0, 100), 0, 0, 0)
        self.assertStepEquals(Step(10, 100), 10, 0, 10)
        self.assertStepEquals(Step(110, 100), 110, 1, 10)
        self.assertStepEquals(Step(11010, 100), 11010, 110, 10)

    def test_from_iteration(self):
        self.assertStepEquals(Step.from_iteration(0, 1), 0, 0, 0)
        self.assertStepEquals(Step.from_iteration(0, 100), 0, 0, 0)
        self.assertStepEquals(Step.from_iteration(10, 100), 10, 0, 10)
        self.assertStepEquals(Step.from_iteration(110, 100), 110, 1, 10)
        self.assertStepEquals(Step.from_iteration(11010, 100), 11010, 110, 10)

    def test_from_epoch(self):
        self.assertStepEquals(Step.from_epoch(0, 0, 1), 0, 0, 0)
        self.assertStepEquals(Step.from_epoch(5, 0, 1), 5, 5, 0)
        self.assertStepEquals(Step.from_epoch(100, 0, 1), 100, 100, 0)
        self.assertStepEquals(Step.from_epoch(100, 20, 1), 120, 120, 0)

        self.assertStepEquals(Step.from_epoch(0, 0, 100), 0, 0, 0)
        self.assertStepEquals(Step.from_epoch(0, 50, 100), 50, 0, 50)
        self.assertStepEquals(Step.from_epoch(1, 0, 100), 100, 1, 0)
        self.assertStepEquals(Step.from_epoch(1, 50, 100), 150, 1, 50)
        self.assertStepEquals(Step.from_epoch(100, 30, 100), 10030, 100, 30)
        self.assertStepEquals(Step.from_epoch(100, 1000, 100), 11000, 110, 0)

    def test_from_str(self):
        self.assertStepEquals(Step.from_str('0it', 100), 0, 0, 0)
        self.assertStepEquals(Step.from_str('0ep', 100), 0, 0, 0)
        self.assertStepEquals(Step.from_str('0ep0it', 100), 0, 0, 0)

        self.assertStepEquals(Step.from_str('50it', 100), 50, 0, 50)
        self.assertStepEquals(Step.from_str('100it', 100), 100, 1, 0)
        self.assertStepEquals(Step.from_str('2021it', 100), 2021, 20, 21)

        self.assertStepEquals(Step.from_str('5ep', 100), 500, 5, 0)
        self.assertStepEquals(Step.from_str('20ep', 100), 2000, 20, 0)

        self.assertStepEquals(Step.from_str('5ep3it', 100), 503, 5, 3)

        with self.assertRaises(ValueError):
            Step.from_str('', 100)

        with self.assertRaises(ValueError):
            Step.from_str('0', 100)

        with self.assertRaises(ValueError):
            Step.from_str('it', 100)

        with self.assertRaises(ValueError):
            Step.from_str('ep', 100)

        with self.assertRaises(ValueError):
            Step.from_str('ep0', 100)

        with self.assertRaises(ValueError):
            Step.from_str('20it50ep', 100)

    def test_zero(self):
        self.assertStepEquals(Step.zero(100), 0, 0, 0)

    def test_equal(self):
        self.assertEqual(Step.from_str('100it', 100), Step.from_str('100it', 100))
        self.assertNotEqual(Step.from_str('101it', 100), Step.from_str('100it', 100))
        self.assertEqual(Step.from_str('1ep', 100), Step.from_str('100it', 100))
        self.assertEqual(Step.from_str('5ep6it', 100), Step.from_str('506it', 100))

        with self.assertRaises(ValueError):
            Step.from_str('100it', 101) == Step.from_str('100it', 100)

    def test_comparisons(self):
        self.assertLessEqual(Step.from_str('100it', 100), Step.from_str('100it', 100))
        self.assertLess(Step.from_str('100it', 100), Step.from_str('101it', 100))
        self.assertLessEqual(Step.from_str('100it', 100), Step.from_str('101it', 100))

        self.assertGreaterEqual(Step.from_str('100it', 100), Step.from_str('100it', 100))
        self.assertGreater(Step.from_str('102it', 100), Step.from_str('101it', 100))
        self.assertGreaterEqual(Step.from_str('102it', 100), Step.from_str('101it', 100))


test_case.main()
