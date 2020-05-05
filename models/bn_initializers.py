# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def uniform(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.rand(w.weight.data.shape)
        w.bias.data = torch.zeros_like(w.bias.data)


def fixed(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.ones_like(w.weight.data)
        w.bias.data = torch.zeros_like(w.bias.data)


def oneone(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.ones_like(w.weight.data)
        w.bias.data = torch.ones_like(w.bias.data)


def positivenegative(w):
    if isinstance(w, torch.nn.BatchNorm2d):
        uniform(w)
        w.weight.data = w.weight.data * 2 - 1
        w.bias.data = torch.zeros_like(w.bias.data)
