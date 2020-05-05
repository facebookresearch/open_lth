# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train


class Branch(base.Branch):
    def branch_function(self, start_at_step_zero: bool = False):
        model = PrunedModel(models.registry.get(self.lottery_desc.model_hparams), Mask.load(self.level_root))
        start_step = self.lottery_desc.str_to_step('0it') if start_at_step_zero else self.lottery_desc.train_start_step
        Mask.load(self.level_root).save(self.branch_root)
        train.standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Randomly reinitialize the model."

    @staticmethod
    def name():
        return 'randomly_reinitialize'
