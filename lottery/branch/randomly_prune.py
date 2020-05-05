# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import vectorize, unvectorize, shuffle_tensor, shuffle_state_dict


class Branch(base.Branch):
    def branch_function(self, seed: int, strategy: str = 'layerwise', start_at: str = 'rewind',
                        layers_to_ignore: str = ''):
        # Randomize the mask.
        mask = Mask.load(self.level_root)

        # Randomize while keeping the same layerwise proportions as the original mask.
        if strategy == 'layerwise': mask = Mask(shuffle_state_dict(mask, seed=seed))

        # Randomize globally throughout all prunable layers.
        elif strategy == 'global': mask = Mask(unvectorize(shuffle_tensor(vectorize(mask), seed=seed), mask))

        # Randomize evenly across all layers.
        elif strategy == 'even':
            sparsity = mask.sparsity
            for i, k in sorted(mask.keys()):
                layer_mask = torch.where(torch.arange(mask[k].size) < torch.ceil(sparsity * mask[k].size),
                                         torch.ones_like(mask[k].size), torch.zeros_like(mask[k].size))
                mask[k] = shuffle_tensor(layer_mask, seed=seed+i).reshape(mask[k].size)

        # Identity.
        elif strategy == 'identity': pass

        # Error.
        else: raise ValueError(f'Invalid strategy: {strategy}')

        # Reset the masks of any layers that shouldn't be pruned.
        if layers_to_ignore:
            for k in layers_to_ignore.split(','): mask[k] = torch.ones_like(mask[k])

        # Save the new mask.
        mask.save(self.branch_root)

        # Determine the start step.
        if start_at == 'init':
            start_step = self.lottery_desc.str_to_step('0ep')
            state_step = start_step
        elif start_at == 'end':
            start_step = self.lottery_desc.str_to_step('0ep')
            state_step = self.lottery_desc.train_end_step
        elif start_at == 'rewind':
            start_step = self.lottery_desc.train_start_step
            state_step = start_step
        else:
            raise ValueError(f'Invalid starting point {start_at}')

        # Train the model with the new mask.
        model = PrunedModel(models.registry.load(self.level_root, state_step, self.lottery_desc.model_hparams), mask)
        train.standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Randomly prune the model."

    @staticmethod
    def name():
        return 'randomly_prune'
