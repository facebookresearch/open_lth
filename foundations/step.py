# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Step:
    """Represents a particular step of training.

    A step can be represented as either an iteration or a pair of an epoch and an iteration within that epoch.
    This class encapsulates a step of training such that it can be freely converted between the two representations.
    """

    def __init__(self, iteration: int, iterations_per_epoch: int) -> 'Step':
        if iteration < 0: raise ValueError('iteration must >= 0.')
        if iterations_per_epoch <= 0: raise ValueError('iterations_per_epoch must be > 0.')
        self._iteration = iteration
        self._iterations_per_epoch = iterations_per_epoch

    @staticmethod
    def str_is_zero(s: str):
        return s in ['0ep', '0it', '0ep0it']

    @staticmethod
    def from_iteration(iteration: int, iterations_per_epoch: int) -> 'Step':
        return Step(iteration, iterations_per_epoch)

    @staticmethod
    def from_epoch(epoch: int, iteration: int, iterations_per_epoch: int) -> 'Step':
        return Step(epoch * iterations_per_epoch + iteration, iterations_per_epoch)

    @staticmethod
    def from_str(s: str, iterations_per_epoch: int) -> 'Step':
        """Creates a step from a string that describes the number of epochs, iterations, or both.

        Epochs: '120ep'
        Iterations: '2000it'
        Both: '120ep50it'"""

        if 'ep' in s and 'it' in s:
            ep = int(s.split('ep')[0])
            it = int(s.split('ep')[1].split('it')[0])
            if s != '{}ep{}it'.format(ep, it): raise ValueError('Malformed string step: {}'.format(s))
            return Step.from_epoch(ep, it, iterations_per_epoch)
        elif 'ep' in s:
            ep = int(s.split('ep')[0])
            if s != '{}ep'.format(ep): raise ValueError('Malformed string step: {}'.format(s))
            return Step.from_epoch(ep, 0, iterations_per_epoch)
        elif 'it' in s:
            it = int(s.split('it')[0])
            if s != '{}it'.format(it): raise ValueError('Malformed string step: {}'.format(s))
            return Step.from_iteration(it, iterations_per_epoch)
        else:
            raise ValueError('Malformed string step: {}'.format(s))

    @staticmethod
    def zero(iterations_per_epoch: int) -> 'Step':
        return Step(0, iterations_per_epoch)

    @property
    def iteration(self):
        """The overall number of steps of training completed so far."""
        return self._iteration

    @property
    def ep(self):
        """The current epoch of training."""
        return self._iteration // self._iterations_per_epoch

    @property
    def it(self):
        """The iteration within the current epoch of training."""
        return self._iteration % self._iterations_per_epoch

    def _check(self, other):
        if not isinstance(other, Step):
            raise ValueError('Invalid type for other: {}.'.format(type(other)))
        if self._iterations_per_epoch != other._iterations_per_epoch:
            raise ValueError('Cannot compare steps when epochs are of different lengths.')

    def __lt__(self, other):
        self._check(other)
        return self._iteration < other._iteration

    def __le__(self, other):
        self._check(other)
        return self._iteration <= other._iteration

    def __eq__(self, other):
        self._check(other)
        return self._iteration == other._iteration

    def __ne__(self, other):
        self._check(other)
        return self._iteration != other._iteration

    def __gt__(self, other):
        self._check(other)
        return self._iteration > other._iteration

    def __ge__(self, other):
        self._check(other)
        return self._iteration >= other._iteration

    def __str__(self):
        return '(Iteration {}; Iterations per Epoch: {})'.format(self._iteration, self._iterations_per_epoch)
