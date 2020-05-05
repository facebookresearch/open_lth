# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import datasets.registry
from foundations.step import Step
import models.registry
from training import train
from testing import test_case


class TestTrain(test_case.TestCase):
    def setUp(self):
        super(TestTrain, self).setUp()
        self.hparams = models.registry.get_default_hparams('mnist_lenet_10_10')
        self.hparams.dataset_hparams.subsample_fraction = 0.01
        self.hparams.dataset_hparams.batch_size = 50  # Leads to 12 iterations per epoch.
        self.hparams.dataset_hparams.do_not_augment = True
        self.hparams.training_hparams.data_order_seed = 0
        self.train_loader = datasets.registry.get(self.hparams.dataset_hparams)

        self.hparams.training_hparams.training_steps = '3ep'
        self.hparams.training_hparams.warmup_steps = '10it'
        self.hparams.training_hparams.gamma = 0.1
        self.hparams.training_hparams.milestone_steps = '2ep'

        self.model = models.registry.get(self.hparams.model_hparams)

        self.step_counter = 0
        self.ep = 0
        self.it = 0
        self.lr = 0.0

        def callback(output_location, step, model, optimizer, logger):
            self.step_counter += 1
            self.ep, self.it = step.ep, step.it
            self.lr = np.round(optimizer.param_groups[0]['lr'], 10)

        self.callback = callback

    def test_train_zero_steps(self):
        before = TestTrain.get_state(self.model)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    end_step=Step.from_iteration(0, len(self.train_loader)))

        after = TestTrain.get_state(self.model)
        for k in before:
            self.assertTrue(np.array_equal(before[k], after[k]))
        self.assertEqual(self.step_counter, 0)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 0)

    def test_train_two_steps(self):
        before = TestTrain.get_state(self.model)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    end_step=Step.from_iteration(2, len(self.train_loader)))

        after = TestTrain.get_state(self.model)
        for k in before:
            with self.subTest(k=k):
                self.assertFalse(np.array_equal(before[k], after[k]), k)

        self.assertEqual(self.step_counter, 3)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 2)
        self.assertEqual(self.lr, 0.02)

    def test_train_one_epoch(self):
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    end_step=Step.from_epoch(1, 0, len(self.train_loader)))

        self.assertEqual(self.step_counter, 13)  # Same as len(self.train_loader) + 1
        self.assertEqual(self.ep, 1)
        self.assertEqual(self.it, 0)
        self.assertEqual(self.lr, 0.1)

    def test_train_more_than_two_epochs(self):
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    end_step=Step.from_epoch(2, 1, len(self.train_loader)))

        self.assertEqual(self.step_counter, 26)
        self.assertEqual(self.ep, 2)
        self.assertEqual(self.it, 1)
        self.assertEqual(self.lr, 0.01)

    def test_train_in_full(self):
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback])

        self.assertEqual(self.step_counter, 37)
        self.assertEqual(self.ep, 3)
        self.assertEqual(self.it, 0)
        self.assertEqual(self.lr, 0.01)

    def test_train_zero_steps_late_start(self):
        before = TestTrain.get_state(self.model)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(0, 5, len(self.train_loader)),
                    end_step=Step.from_epoch(0, 5, len(self.train_loader)))

        after = TestTrain.get_state(self.model)
        for k in before:
            self.assertTrue(np.array_equal(before[k], after[k]))
        self.assertEqual(self.step_counter, 0)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 0)

    def test_train_one_step_late_start(self):
        before = TestTrain.get_state(self.model)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(0, 5, len(self.train_loader)),
                    end_step=Step.from_epoch(0, 6, len(self.train_loader)))

        after = TestTrain.get_state(self.model)
        for k in before:
            self.assertFalse(np.array_equal(before[k], after[k]))
        self.assertEqual(self.step_counter, 2)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 6)
        self.assertEqual(self.lr, 0.06)

    def test_train_one_epoch_late_start(self):
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(0, 5, len(self.train_loader)),
                    end_step=Step.from_epoch(1, 5, len(self.train_loader)))

        self.assertEqual(self.step_counter, 13)
        self.assertEqual(self.ep, 1)
        self.assertEqual(self.it, 5)
        self.assertEqual(self.lr, 0.1)

    def test_train_two_epoch_late_start(self):
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(0, 5, len(self.train_loader)),
                    end_step=Step.from_epoch(2, 5, len(self.train_loader)))

        self.assertEqual(self.step_counter, 25)
        self.assertEqual(self.ep, 2)
        self.assertEqual(self.it, 5)
        self.assertEqual(self.lr, 0.01)

    def test_train_in_full_late_start(self):
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(0, 5, len(self.train_loader)))

        self.assertEqual(self.step_counter, 32)
        self.assertEqual(self.ep, 3)
        self.assertEqual(self.it, 0)
        self.assertEqual(self.lr, 0.01)

    def test_train_in_full_later_start(self):
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 5, len(self.train_loader)))

        self.assertEqual(self.step_counter, 20)
        self.assertEqual(self.ep, 3)
        self.assertEqual(self.it, 0)
        self.assertEqual(self.lr, 0.01)

    def test_train_in_parts(self):
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    end_step=Step.from_epoch(0, 7, len(self.train_loader)))

        self.assertEqual(self.step_counter, 8)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 7)
        self.assertEqual(self.lr, 0.07)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(0, 7, len(self.train_loader)),
                    end_step=Step.from_epoch(0, 8, len(self.train_loader)))

        self.assertEqual(self.step_counter, 10)
        self.assertEqual(self.ep, 0)
        self.assertEqual(self.it, 8)
        self.assertEqual(self.lr, 0.08)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(0, 8, len(self.train_loader)),
                    end_step=Step.from_epoch(1, 2, len(self.train_loader)))

        self.assertEqual(self.step_counter, 17)
        self.assertEqual(self.ep, 1)
        self.assertEqual(self.it, 2)
        self.assertEqual(self.lr, 0.1)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 2, len(self.train_loader)),
                    end_step=Step.from_epoch(2, 1, len(self.train_loader)))

        self.assertEqual(self.step_counter, 29)
        self.assertEqual(self.ep, 2)
        self.assertEqual(self.it, 1)
        self.assertEqual(self.lr, 0.01)

        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(2, 1, len(self.train_loader)),
                    end_step=Step.from_epoch(3, 0, len(self.train_loader)))

        self.assertEqual(self.step_counter, 41)
        self.assertEqual(self.ep, 3)
        self.assertEqual(self.it, 0)
        self.assertEqual(self.lr, 0.01)

    def test_repeatable_data_order_with_seed(self):
        init = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        # Train the model once and get the state.
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 0, len(self.train_loader)),
                    end_step=Step.from_epoch(1, 1, len(self.train_loader)))
        state1 = TestTrain.get_state(self.model)

        # Train the model again and get the state.
        self.model.load_state_dict(init)
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 0, len(self.train_loader)),
                    end_step=Step.from_epoch(1, 1, len(self.train_loader)))
        state2 = TestTrain.get_state(self.model)

        # Ensure that the model states are the same.
        for k in state1:
            self.assertTrue(np.array_equal(state1[k], state2[k]))

    def test_nonrepeatable_data_order_without_seed(self):
        del self.hparams.training_hparams.data_order_seed

        init = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        # Train the model once and get the state.
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 0, len(self.train_loader)),
                    end_step=Step.from_epoch(1, 1, len(self.train_loader)))
        state1 = TestTrain.get_state(self.model)

        # Train the model again and get the state.
        self.model.load_state_dict(init)
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 0, len(self.train_loader)),
                    end_step=Step.from_epoch(1, 1, len(self.train_loader)))
        state2 = TestTrain.get_state(self.model)

        # Ensure that the model states are NOT the same.
        for k in state1:
            self.assertFalse(np.array_equal(state1[k], state2[k]))

    def test_different_data_on_different_steps(self):
        init = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        # Train the model once and get the state.
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 0, len(self.train_loader)),
                    end_step=Step.from_epoch(1, 1, len(self.train_loader)))
        state1 = TestTrain.get_state(self.model)

        # Train the model again and get the state.
        self.model.load_state_dict(init)
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 1, len(self.train_loader)),
                    end_step=Step.from_epoch(1, 2, len(self.train_loader)))
        state2 = TestTrain.get_state(self.model)

        # Ensure that the model states are NOT the same.
        for k in state1:
            self.assertFalse(np.array_equal(state1[k], state2[k]))

    def test_different_data_order_on_different_epochs(self):
        del self.hparams.training_hparams.gamma
        del self.hparams.training_hparams.milestone_steps
        del self.hparams.training_hparams.warmup_steps

        init = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        # Train the model once and get the state.
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(1, 0, len(self.train_loader)),
                    end_step=Step.from_epoch(1, 1, len(self.train_loader)))
        state1 = TestTrain.get_state(self.model)

        # Train the model again and get the state.
        self.model.load_state_dict(init)
        train.train(self.hparams.training_hparams, self.model, self.train_loader,
                    self.root, callbacks=[self.callback],
                    start_step=Step.from_epoch(2, 0, len(self.train_loader)),
                    end_step=Step.from_epoch(2, 1, len(self.train_loader)))
        state2 = TestTrain.get_state(self.model)

        # Ensure that the model states are NOT the same.
        for k in state1:
            self.assertFalse(np.array_equal(state1[k], state2[k]))


test_case.main()
