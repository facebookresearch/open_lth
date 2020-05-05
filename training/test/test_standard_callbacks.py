# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

import datasets.registry
from foundations import paths
from foundations.step import Step
import models.registry
from training import standard_callbacks
from training import train
from training.metric_logger import MetricLogger
from testing import test_case


class TestStandardCallbacks(test_case.TestCase):
    def setUp(self):
        super(TestStandardCallbacks, self).setUp()

        # Model hparams.
        self.hparams = models.registry.get_default_hparams('mnist_lenet_10_10')
        self.model = models.registry.get(self.hparams.model_hparams)

        # Dataset hparams.
        self.hparams.dataset_hparams.subsample_fraction = 0.01
        self.hparams.dataset_hparams.batch_size = 50
        self.train_loader = datasets.registry.get(self.hparams.dataset_hparams)
        self.test_loader = datasets.registry.get(self.hparams.dataset_hparams, train=False)

        # Training hparams.
        self.hparams.training_hparams.training_steps = '3ep'

        # Get the callbacks.
        self.callbacks = standard_callbacks.standard_callbacks(
            self.hparams.training_hparams, self.train_loader, self.test_loader,
            eval_on_train=True, verbose=False)

    def test_first_step(self):
        init_state = TestStandardCallbacks.get_state(self.model)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=self.callbacks,
                    end_step=Step.from_epoch(0, 1, len(self.train_loader)))

        # Check that the initial state has been saved.
        model_state_loc = paths.model(self.root, Step.zero(len(self.train_loader)))
        self.assertTrue(os.path.exists(model_state_loc))

        # Check that the model state at init reflects the saved state.
        self.model.load_state_dict(torch.load(model_state_loc))
        saved_state = TestStandardCallbacks.get_state(self.model)
        self.assertStateEqual(init_state, saved_state)

        # Check that the checkpoint file exists.
        self.assertTrue(os.path.exists(paths.checkpoint(self.root)))

        # Check that the logger file doesn't exist.
        self.assertFalse(os.path.exists(paths.logger(self.root)))

    def test_last_step(self):
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=self.callbacks,
                    start_step=Step.from_epoch(2, 11, len(self.train_loader)),
                    end_step=Step.from_epoch(3, 0, len(self.train_loader)))

        end_state = TestStandardCallbacks.get_state(self.model)

        # Check that final state has been saved.
        end_loc = paths.model(self.root, Step.from_epoch(3, 0, len(self.train_loader)))
        self.assertTrue(os.path.exists(end_loc))

        # Check that the final state that is saved matches the final state of the network.
        self.model.load_state_dict(torch.load(end_loc))
        saved_state = TestStandardCallbacks.get_state(self.model)
        self.assertStateEqual(end_state, saved_state)

        # Check that the logger has the right number of entries.
        self.assertTrue(os.path.exists(paths.logger(self.root)))
        logger = MetricLogger.create_from_file(self.root)
        self.assertEqual(len(logger.get_data('train_loss')), 1)
        self.assertEqual(len(logger.get_data('test_loss')), 1)
        self.assertEqual(len(logger.get_data('train_accuracy')), 1)
        self.assertEqual(len(logger.get_data('test_accuracy')), 1)

        # Check that the checkpoint file exists.
        self.assertTrue(os.path.exists(paths.checkpoint(self.root)))

    def test_end_to_end(self):
        init_loc = paths.model(self.root, Step.zero(len(self.train_loader)))
        end_loc = paths.model(self.root, Step.from_epoch(3, 0, len(self.train_loader)))

        init_state = TestStandardCallbacks.get_state(self.model)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=self.callbacks,
                    start_step=Step.from_epoch(0, 0, len(self.train_loader)),
                    end_step=Step.from_epoch(3, 0, len(self.train_loader)))

        end_state = TestStandardCallbacks.get_state(self.model)

        # Check that final state has been saved.
        self.assertTrue(os.path.exists(init_loc))
        self.assertTrue(os.path.exists(end_loc))

        # Check that the checkpoint file still exists.
        self.assertTrue(os.path.exists(paths.checkpoint(self.root)))

        # Check that the initial and final states match those that were saved.
        self.model.load_state_dict(torch.load(init_loc))
        saved_state = TestStandardCallbacks.get_state(self.model)
        self.assertStateEqual(init_state, saved_state)

        self.model.load_state_dict(torch.load(end_loc))
        saved_state = TestStandardCallbacks.get_state(self.model)
        self.assertStateEqual(end_state, saved_state)

        # Check that the logger has the right number of entries.
        self.assertTrue(os.path.exists(paths.logger(self.root)))
        logger = MetricLogger.create_from_file(self.root)
        self.assertEqual(len(logger.get_data('train_loss')), 4)
        self.assertEqual(len(logger.get_data('test_loss')), 4)
        self.assertEqual(len(logger.get_data('train_accuracy')), 4)
        self.assertEqual(len(logger.get_data('test_accuracy')), 4)

    def test_checkpointing(self):
        callback_step_count = 0

        def callback(output_location, step, model, optimizer, logger):
            nonlocal callback_step_count
            callback_step_count += 1

        # Train to epoch 1, iteration 1.
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=self.callbacks,
                    end_step=Step.from_epoch(1, 1, len(self.train_loader)))

        # Add a step-counting callback.
        self.callbacks.append(callback)

        # Train to epoch 1, iteration 1 again. Checkpointing should ensure we
        # only train for one step.
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=self.callbacks,
                    end_step=Step.from_epoch(1, 1, len(self.train_loader)))

        self.assertEqual(callback_step_count, 2)


test_case.main()
