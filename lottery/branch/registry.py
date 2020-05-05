# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lottery.branch.base import Branch
from lottery.branch import randomly_prune
from lottery.branch import randomly_reinitialize
from lottery.branch import retrain

registered_branches = {
    'randomly_prune': randomly_prune.Branch,
    'randomly_reinitialize': randomly_reinitialize.Branch,
    'retrain': retrain.Branch,
}


def get(branch_name: str) -> Branch:
    if branch_name not in registered_branches:
        raise ValueError('No such branch: {}'.format(branch_name))
    else:
        return registered_branches[branch_name]
