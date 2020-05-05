# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import argparse
from dataclasses import dataclass, field, make_dataclass, fields
import inspect
from typing import List

from cli import shared_args
from foundations.desc import Desc
from foundations.hparams import Hparams
from foundations.runner import Runner
from lottery.desc import LotteryDesc
from lottery.branch.desc import make_BranchDesc
from platforms.platform import get_platform


@dataclass
class Branch(Runner):
    """A lottery branch. Implement `branch_function`, add a name and description, and add to the registry."""

    replicate: int
    levels: str
    desc: Desc
    verbose: bool = False
    level: int = None

    # Interface that needs to be overriden for each branch.
    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """A description of this branch. Override this."""
        pass

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """The name of this branch. Override this."""
        pass

    @abc.abstractmethod
    def branch_function(self) -> None:
        """The method that is called to execute the branch.

        Override this method with any additional arguments that the branch will need.
        These arguments will be converted into command-line arguments for the branch.
        Each argument MUST have a type annotation. The first argument must still be self.
        """
        pass

    # Interface that is useful for writing branches.
    @property
    def lottery_desc(self) -> LotteryDesc:
        """The lottery description of this experiment."""

        return self.desc.lottery_desc

    @property
    def experiment_name(self) -> str:
        """The name of this experiment."""

        return self.desc.hashname

    @property
    def branch_root(self) -> str:
        """The root for where branch results will be stored for a specific invocation of run()."""

        return self.lottery_desc.run_path(self.replicate, self.level, self.experiment_name)

    @property
    def level_root(self) -> str:
        """The root of the main experiment on which this branch is based."""

        return self.lottery_desc.run_path(self.replicate, self.level)

    # Interface that deals with command line arguments.
    @dataclass
    class ArgHparams(Hparams):
        levels: str
        pretrain_training_steps: str = None

        _name: str = 'Lottery Ticket Hyperparameters'
        _description: str = 'Hyperparameters that control the lottery ticket process.'
        _levels: str = \
            'The pruning levels on which to run this branch. Can include a comma-separate list of levels or ranges, '\
            'e.g., 1,2-4,9'
        _pretrain_training_steps: str = 'The number of steps to train the network prior to the lottery ticket process.'

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        defaults = shared_args.maybe_get_default_hparams()
        shared_args.JobArgs.add_args(parser)
        Branch.ArgHparams.add_args(parser)
        cls.BranchDesc.add_args(parser, defaults)

    @staticmethod
    def level_str_to_int_list(levels: str):
        level_list = []
        elements = levels.split(',')
        for element in elements:
            if element.isdigit():
                level_list.append(int(element))
            elif len(element.split('-')) == 2:
                level_list += list(range(int(element.split('-')[0]), int(element.split('-')[1]) + 1))
            else:
                raise ValueError(f'Invalid level: {element}')
        return sorted(list(set(level_list)))

    @classmethod
    def create_from_args(cls, args: argparse.Namespace):
        levels = Branch.level_str_to_int_list(args.levels)
        return cls(args.replicate, levels, cls.BranchDesc.create_from_args(args), not args.quiet)

    @classmethod
    def create_from_hparams(cls, replicate, levels: List[int], desc: LotteryDesc, hparams: Hparams, verbose=False):
        return cls(replicate, levels, cls.BranchDesc(desc, hparams), verbose)

    def display_output_location(self):
        print(self.branch_root)

    def run(self):
        for self.level in self.levels:
            if self.verbose and get_platform().is_primary_process:
                print('='*82)
                print(f'Branch {self.name()} (Replicate {self.replicate}, Level {self.level})\n' + '-'*82)
                print(f'{self.lottery_desc.display}\n{self.desc.branch_hparams.display}')
                print(f'Output Location: {self.branch_root}\n' + '='*82 + '\n')

            args = {f.name: getattr(self.desc.branch_hparams, f.name)
                    for f in fields(self.BranchHparams) if not f.name.startswith('_')}
            self.branch_function(**args)

    # Initialize instances and subclasses (metaprogramming).
    def __init_subclass__(cls):
        """Metaprogramming: modify the attributes of the subclass based on information in run().

        The goal is to make it possible for users to simply write a single run() method and have
        as much functionality as possible occur automatically. Specifically, this function converts
        the annotations and defaults in run() into a `BranchHparams` property.
        """

        fields = []
        for arg_name, parameter in list(inspect.signature(cls.branch_function).parameters.items())[1:]:
            t = parameter.annotation
            if t == inspect._empty: raise ValueError(f'Argument {arg_name} needs a type annotation.')
            elif t in [str, float, int, bool] or (isinstance(t, type) and issubclass(t, Hparams)):
                if parameter.default != inspect._empty: fields.append((arg_name, t, field(default=parameter.default)))
                else: fields.append((arg_name, t))
            else:
                raise ValueError('Invalid branch type: {}'.format(parameter.annotation))

        fields += [('_name', str, 'Branch Arguments'), ('_description', str, 'Arguments specific to the branch.')]
        setattr(cls, 'BranchHparams', make_dataclass('BranchHparams', fields, bases=(Hparams,)))
        setattr(cls, 'BranchDesc', make_BranchDesc(cls.BranchHparams, cls.name()))
