# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os


def checkpoint(root): return os.path.join(root, 'checkpoint.pth')


def logger(root): return os.path.join(root, 'logger')


def mask(root): return os.path.join(root, 'mask.pth')


def sparsity_report(root): return os.path.join(root, 'sparsity_report.json')


def model(root, step): return os.path.join(root, 'model_ep{}_it{}.pth'.format(step.ep, step.it))


def hparams(root): return os.path.join(root, 'hparams.log')
