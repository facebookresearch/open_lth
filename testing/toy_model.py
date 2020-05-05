# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.base import Model


class InnerProductModel(Model):
    @staticmethod
    def default_hparams(): raise NotImplementedError

    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError

    @staticmethod
    def get_model_from_name(model_name): raise NotImplementedError

    @property
    def output_layer_names(self): raise NotImplementedError

    @property
    def loss_criterion(self): return torch.nn.MSELoss()

    def __init__(self, n):
        super(Model, self).__init__()
        self.layer = torch.nn.Linear(n, 1, bias=False)
        self.layer.weight.data = torch.arange(n, dtype=torch.float32)

    def forward(self, x):
        return self.layer(x)
