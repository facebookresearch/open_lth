# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from platforms import local


registered_platforms = {'local': local.Platform}


def get(name):
    return registered_platforms[name]
