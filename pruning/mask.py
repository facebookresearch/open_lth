# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import numpy as np
import torch

from foundations import paths
from platforms.platform import get_platform

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models import base



class Mask(dict):
    def __init__(self, other_dict=None):
        super(Mask, self).__init__()
        if other_dict is not None:
            for k, v in other_dict.items(): self[k] = v

    def __setitem__(self, key, value):
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError('Invalid tensor name: {}'.format(key))
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError('value for key {} must be torch Tensor or numpy ndarray.'.format(key))
        if ((value != 0) & (value != 1)).any(): raise ValueError('All entries must be 0 or 1.')

        super(Mask, self).__setitem__(key, value)

    @staticmethod
    def ones_like(model: 'base.Model') -> 'Mask':
        mask = Mask()
        for name in model.prunable_layer_names:
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    def save(self, output_location):
        if not get_platform().is_primary_process: return
        if not get_platform().exists(output_location): get_platform().makedirs(output_location)
        get_platform().save_model({k: v.cpu().int() for k, v in self.items()}, paths.mask(output_location))

        # Create a sparsity report.
        total_weights = np.sum([v.size for v in self.numpy().values()]).item()
        total_unpruned = np.sum([np.sum(v) for v in self.numpy().values()]).item()
        with get_platform().open(paths.sparsity_report(output_location), 'w') as fp:
            fp.write(json.dumps({'total': float(total_weights), 'unpruned': float(total_unpruned)}, indent=4))

    @staticmethod
    def load(output_location):
        if not Mask.exists(output_location):
            raise ValueError('Mask not found at {}'.format(output_location))
        return Mask(get_platform().load_model(paths.mask(output_location)))

    @staticmethod
    def exists(output_location):
        return get_platform().exists(paths.mask(output_location))

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity
