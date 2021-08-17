# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
from dataclasses import dataclass, replace
import os
from typing import Union

from cli import arg_utils
from datasets import registry as datasets_registry
from foundations.desc import Desc
from foundations import hparams
from foundations.step import Step
from platforms.platform import get_platform
import pruning.registry


@dataclass
class LotteryDesc(Desc):
    """The hyperparameters necessary to describe a lottery ticket training backbone."""

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    pruning_hparams: hparams.PruningHparams
    pretrain_dataset_hparams: hparams.DatasetHparams = None
    pretrain_training_hparams: hparams.TrainingHparams = None

    @staticmethod
    def name_prefix(): return 'lottery'

    @staticmethod
    def _add_pretrain_argument(parser):
        help_text = \
            'Perform a pre-training phase prior to running the main lottery ticket process. Setting this argument '\
            'will enable arguments to control how the dataset and training during this pre-training phase. Rewinding '\
            'is a specific case of of pre-training where pre-training uses the same dataset and training procedure '\
            'as the main training run.'
        parser.add_argument('--pretrain', action='store_true', help=help_text)

    @staticmethod
    def _add_rewinding_argument(parser):
        help_text = \
            'The number of steps for which to train the network before the lottery ticket process begins. This is ' \
            'the \'rewinding\' step as described in recent lottery ticket research. Can be expressed as a number of ' \
            'epochs (\'160ep\') or a number  of iterations (\'50000it\'). If this flag is present, no other '\
            'pretraining arguments  may be set. Pretraining will be conducted using the same dataset and training '\
            'hyperparameters as for the main training run. For the full range of pre-training options, use --pretrain.'
        parser.add_argument('--rewinding_steps', type=str, help=help_text)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'LotteryDesc' = None):
        # Add the rewinding/pretraining arguments.
        rewinding_steps = arg_utils.maybe_get_arg('rewinding_steps')
        pretrain = arg_utils.maybe_get_arg('pretrain', boolean_arg=True)

        if rewinding_steps is not None and pretrain: raise ValueError('Cannot set --rewinding_steps and --pretrain')
        pretraining_parser = parser.add_argument_group(
            'Rewinding/Pretraining Arguments', 'Arguments that control how the network is pre-trained')
        LotteryDesc._add_rewinding_argument(pretraining_parser)
        LotteryDesc._add_pretrain_argument(pretraining_parser)

        # Get the proper pruning hparams.
        pruning_strategy = arg_utils.maybe_get_arg('pruning_strategy')
        if defaults and not pruning_strategy: pruning_strategy = defaults.pruning_hparams.pruning_strategy
        def_ph = None
        if pruning_strategy:
            pruning_hparams = pruning.registry.get_pruning_hparams(pruning_strategy)
            if defaults and defaults.pruning_hparams.pruning_strategy == pruning_strategy:
                def_ph = defaults.pruning_hparams
        else:
            pruning_hparams = hparams.PruningHparams

        # Add the main arguments.
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        pruning_hparams.add_args(parser, defaults=def_ph if defaults else None)

        # Handle pretraining.
        if pretrain:
            if defaults: def_th = replace(defaults.training_hparams, training_steps='0ep')
            hparams.TrainingHparams.add_args(parser, defaults=def_th if defaults else None,
                                             name='Training Hyperparameters for Pretraining', prefix='pretrain')
            hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None,
                                            name='Dataset Hyperparameters for Pretraining', prefix='pretrain')

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> 'LotteryDesc':
        # Get the main arguments.
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        pruning_hparams = pruning.registry.get_pruning_hparams(args.pruning_strategy).create_from_args(args)

        # Create the desc.
        desc = cls(model_hparams, dataset_hparams, training_hparams, pruning_hparams)

        # Handle pretraining.
        if args.pretrain and not Step.str_is_zero(args.pretrain_training_steps):
            desc.pretrain_dataset_hparams = hparams.DatasetHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_dataset_hparams._name = 'Pretraining ' + desc.pretrain_dataset_hparams._name
            desc.pretrain_training_hparams = hparams.TrainingHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_training_hparams._name = 'Pretraining ' + desc.pretrain_training_hparams._name
        elif 'rewinding_steps' in args and args.rewinding_steps and not Step.str_is_zero(args.rewinding_steps):
            desc.pretrain_dataset_hparams = copy.deepcopy(dataset_hparams)
            desc.pretrain_dataset_hparams._name = 'Pretraining ' + desc.pretrain_dataset_hparams._name
            desc.pretrain_training_hparams = copy.deepcopy(training_hparams)
            desc.pretrain_training_hparams._name = 'Pretraining ' + desc.pretrain_training_hparams._name
            desc.pretrain_training_hparams.training_steps = args.rewinding_steps

        return desc

    def str_to_step(self, s: str, pretrain: bool = False) -> Step:
        dataset_hparams = self.pretrain_dataset_hparams if pretrain else self.dataset_hparams
        iterations_per_epoch = datasets_registry.iterations_per_epoch(dataset_hparams)
        return Step.from_str(s, iterations_per_epoch)

    @property
    def pretrain_end_step(self):
        return self.str_to_step(self.pretrain_training_hparams.training_steps, True)

    @property
    def train_start_step(self):
        if self.pretrain_training_hparams: return self.str_to_step(self.pretrain_training_hparams.training_steps)
        else: return self.str_to_step('0it')

    @property
    def train_end_step(self):
        return self.str_to_step(self.training_hparams.training_steps)

    @property
    def pretrain_outputs(self):
        datasets_registry.num_classes(self.pretrain_dataset_hparams)

    @property
    def train_outputs(self):
        datasets_registry.num_classes(self.dataset_hparams)

    def run_path(self, replicate: int, pruning_level: Union[str, int], experiment: str = 'main'):
        """The location where any run is stored."""

        if not isinstance(replicate, int) or replicate <= 0:
            raise ValueError('Bad replicate: {}'.format(replicate))

        return os.path.join(get_platform().root, self.hashname,
                            f'replicate_{replicate}', f'level_{pruning_level}', experiment)

    @property
    def display(self):
        ls = [self.dataset_hparams.display, self.model_hparams.display,
              self.training_hparams.display, self.pruning_hparams.display]
        if self.pretrain_training_hparams:
            ls += [self.pretrain_dataset_hparams.display, self.pretrain_training_hparams.display]
        return '\n'.join(ls)
