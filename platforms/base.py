# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass
import os
import torch

from foundations.hparams import Hparams
import platforms.platform


@dataclass
class Platform(Hparams):
    num_workers: int = 0

    _name: str = 'Platform Hyperparameters'
    _description: str = 'Hyperparameters that control the plaform on which the job is run.'
    _num_workers: str = 'The number of worker threads to use for data loading.'

    # Manage the available devices and the status of distributed training.

    @property
    def device_str(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def torch_device(self):
        return torch.device(self.device_str)

    @property
    def is_parallel(self):
        return torch.cuda.is_available() and torch.cuda.device_count() > 1

    @property
    def is_distributed(self):
        return False

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    @property
    def is_primary_process(self):
        return not self.is_distributed or self.rank == 0

    def barrier(self):
        pass

    # Manage the location of important files.

    @property
    @abc.abstractmethod
    def root(self):
        """The root directory where data will be stored."""
        pass

    @property
    @abc.abstractmethod
    def dataset_root(self):
        """The root directory where datasets will be stored."""
        pass

    @property
    @abc.abstractmethod
    def imagenet_root(self):
        """The directory where imagenet will be stored."""
        pass

    # Mediate access to files.
    @staticmethod
    def open(file, mode='r'):
        return open(file, mode)

    @staticmethod
    def exists(file):
        return os.path.exists(file)

    @staticmethod
    def makedirs(path):
        return os.makedirs(path)

    @staticmethod
    def isdir(path):
        return os.path.isdir(path)

    @staticmethod
    def listdir(path):
        return os.listdir(path)

    @staticmethod
    def save_model(model, path, *args, **kwargs):
        return torch.save(model, path, *args, **kwargs)

    @staticmethod
    def load_model(path, *args, **kwargs):
        return torch.load(path, *args, **kwargs)

    # Run jobs. Called by the command line interface.

    def run_job(self, f):
        """Run a function that trains a network."""
        old_platform = platforms.platform._PLATFORM
        platforms.platform._PLATFORM = self
        f()
        platforms.platform._PLATFORM = old_platform
