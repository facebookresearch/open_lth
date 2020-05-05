# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os

import datasets.registry
from foundations import paths
from foundations.step import Step
import models.registry
from testing import test_case
from training import checkpointing
from training import optimizers
from training.metric_logger import MetricLogger


class TestCheckpointing(test_case.TestCase):
    def test_create_restore_delete(self):
        # Create the hyperparameters and objects to save.
        hp = models.registry.get_default_hparams('cifar_resnet_20')
        model = models.registry.get(hp.model_hparams)
        optimizer = optimizers.get_optimizer(hp.training_hparams, model)
        dataloader = datasets.registry.get(hp.dataset_hparams)
        step = Step.from_epoch(13, 27, 400)

        # Run one step of SGD.
        examples, labels = next(iter(dataloader))
        optimizer.zero_grad()
        model.train()
        model.loss_criterion(model(examples), labels).backward()
        optimizer.step()

        # Create a fake logger.
        logger = MetricLogger()
        logger.add('test_accuracy', Step.from_epoch(0, 0, 400), 0.1)
        logger.add('test_accuracy', Step.from_epoch(10, 0, 400), 0.5)
        logger.add('test_accuracy', Step.from_epoch(100, 0, 400), 0.8)

        # Save a checkpoint.
        checkpointing.save_checkpoint_callback(self.root, step, model, optimizer, logger)
        self.assertTrue(os.path.exists(paths.checkpoint(self.root)))

        # Create new models.
        model2 = models.registry.get(hp.model_hparams)
        optimizer2 = optimizers.get_optimizer(hp.training_hparams, model)

        # Ensure the new model has different weights.
        sd1, sd2 = model.state_dict(), model2.state_dict()
        for k in model.prunable_layer_names:
            self.assertFalse(np.array_equal(sd1[k].numpy(), sd2[k].numpy()))

        self.assertIn('momentum_buffer', optimizer.state[optimizer.param_groups[0]['params'][0]])
        self.assertNotIn('momentum_buffer', optimizer2.state[optimizer.param_groups[0]['params'][0]])

        # Restore the checkpointt.
        step2, logger2 = checkpointing.restore_checkpoint(self.root, model2, optimizer2, 400)

        self.assertTrue(os.path.exists(paths.checkpoint(self.root)))
        self.assertEqual(step, step2)
        self.assertEqual(str(logger), str(logger2))

        # Ensure the new model is now the same.
        sd1, sd2 = model.state_dict(), model2.state_dict()
        self.assertEqual(set(sd1.keys()), set(sd2.keys()))
        for k in sd1:
            self.assertTrue(np.array_equal(sd1[k].numpy(), sd2[k].numpy()))

        # Ensure the new optimizer is now the same.
        mom1 = optimizer.state[optimizer.param_groups[0]['params'][0]]['momentum_buffer']
        mom2 = optimizer2.state[optimizer.param_groups[0]['params'][0]]['momentum_buffer']
        self.assertTrue(np.array_equal(mom1.numpy(), mom2.numpy()))


test_case.main()
