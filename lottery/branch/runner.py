# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
import sys

from cli import arg_utils
from foundations.runner import Runner
from lottery.branch import registry


@dataclass
class BranchRunner(Runner):
    """A meta-runner that calls the branch-specific runner."""

    runner: Runner

    @staticmethod
    def description():
        return "Run a lottery branch."

    @staticmethod
    def add_args(parser):
        # Produce help text for selecting the branch.
        branch_names = sorted(registry.registered_branches.keys())
        helptext = '='*82 + '\nOpenLTH: A Library for Research on Lottery Tickets and Beyond\n' + '-'*82
        helptext += '\nChoose a branch to run:'
        for branch_name in branch_names:
            helptext += "\n    * {} {} {} [...] => {}".format(
                        sys.argv[0], sys.argv[1], branch_name, registry.get(branch_name).description())
        helptext += '\n' + '='*82

        # Print an error message if appropriate.
        branch_name = arg_utils.maybe_get_arg('subcommand', positional=True, position=1)
        if branch_name not in branch_names:
            print(helptext)
            sys.exit(1)

        # Add the arguments for the branch.
        parser.add_argument('branch_name', type=str)
        registry.get(branch_name).add_args(parser)

    @staticmethod
    def create_from_args(args: argparse.Namespace):
        return BranchRunner(registry.get(sys.argv[2]).create_from_args(args))

    def display_output_location(self):
        self.runner.display_output_location()

    def run(self) -> None:
        self.runner.run()
