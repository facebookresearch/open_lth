# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datasets.registry
from foundations import hparams
from foundations.step import Step
from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train


class Branch(base.Branch):
    def branch_function(
        self,
        retrain_d: hparams.DatasetHparams,
        retrain_t: hparams.TrainingHparams,
        start_at_step_zero: bool = False
    ):
        # Get the mask and model.
        m = models.registry.load(self.level_root, self.lottery_desc.train_start_step, self.lottery_desc.model_hparams)
        m = PrunedModel(m, Mask.load(self.level_root))
        start_step = Step.from_iteration(0 if start_at_step_zero else self.lottery_desc.train_start_step.iteration,
                                         datasets.registry.iterations_per_epoch(retrain_d))
        train.standard_train(m, self.branch_root, retrain_d, retrain_t, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Retrain the model with different hyperparameters."

    @staticmethod
    def name():
        return 'retrain'
