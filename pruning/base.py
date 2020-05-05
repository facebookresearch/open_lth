# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

from foundations.hparams import PruningHparams
from models import base
from pruning.mask import Mask


class Strategy(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_pruning_hparams() -> type:
        pass

    @staticmethod
    @abc.abstractmethod
    def prune(pruning_hparams: PruningHparams, trained_model: base.Model, current_mask: Mask = None) -> Mask:
        pass
