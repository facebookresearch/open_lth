# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from foundations.step import Step
from training.metric_logger import MetricLogger
from testing import test_case


class TestMetricLogger(test_case.TestCase):
    def test_create(self):
        MetricLogger()

    @staticmethod
    def create_logger():
        logger = MetricLogger()
        logger.add('train_accuracy', Step.from_iteration(0, 400), 0.5)
        logger.add('train_accuracy', Step.from_iteration(1, 400), 0.6)
        logger.add('test_accuracy',  Step.from_iteration(0, 400), 0.4)
        return logger

    def test_add_get(self):
        logger = TestMetricLogger.create_logger()
        self.assertEqual(logger.get_data('train_accuracy'), [(0, 0.5), (1, 0.6)])
        self.assertEqual(logger.get_data('test_accuracy'), [(0, 0.4)])
        self.assertEqual(logger.get_data('test_loss'), [])

    def test_overwrite(self):
        logger = TestMetricLogger.create_logger()
        logger.add('train_accuracy', Step.from_iteration(0, 400), 1.0)
        self.assertEqual(logger.get_data('train_accuracy'), [(0, 1.0), (1, 0.6)])

    def test_sorting(self):
        logger = TestMetricLogger.create_logger()
        logger.add('train_accuracy', Step.from_iteration(5, 400), 0.9)
        logger.add('train_accuracy', Step.from_iteration(3, 400), 0.7)
        logger.add('train_accuracy', Step.from_iteration(4, 400), 0.8)
        self.assertEqual(logger.get_data('train_accuracy'),
                         [(0, 0.5), (1, 0.6), (3, 0.7), (4, 0.8), (5, 0.9)])

    def test_str(self):
        logger = TestMetricLogger.create_logger()
        expected = ['train_accuracy,0,0.5', 'train_accuracy,1,0.6', 'test_accuracy,0,0.4']
        self.assertEqual(str(logger), '\n'.join(expected))

    def test_create_from_string(self):
        logger = TestMetricLogger.create_logger()
        logger2 = MetricLogger.create_from_string(str(logger))
        self.assertEqual(logger.get_data('train_accuracy'), logger2.get_data('train_accuracy'))
        self.assertEqual(logger.get_data('test_accuracy'), logger2.get_data('test_accuracy'))
        self.assertEqual(str(logger), str(logger2))

    def test_file_operations(self):
        logger = TestMetricLogger.create_logger()
        save_loc = os.path.join(self.root, 'temp_logger')

        logger.save(save_loc)
        logger2 = MetricLogger.create_from_file(save_loc)

        self.assertEqual(logger.get_data('train_accuracy'), logger2.get_data('train_accuracy'))
        self.assertEqual(logger.get_data('test_accuracy'), logger2.get_data('test_accuracy'))
        self.assertEqual(str(logger), str(logger2))


test_case.main()
