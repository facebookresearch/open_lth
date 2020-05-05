# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch

from datasets.base import DataLoader
from foundations import hparams
from foundations.step import Step
from platforms.platform import get_platform
from training import checkpointing


# Standard callbacks.
def save_model(output_location, step, model, optimizer, logger):
    model.save(output_location, step)


def save_logger(output_location, step, model, optimizer, logger):
    logger.save(output_location)


def create_timekeeper_callback():
    time_of_last_call = None

    def callback(output_location, step, model, optimizer, logger):
        if get_platform().is_primary_process:
            nonlocal time_of_last_call
            t = 0.0 if time_of_last_call is None else time.time() - time_of_last_call
            print(f'Ep {step.ep}\tIt {step.it}\tTime Elapsed {t:.2f}')
            time_of_last_call = time.time()
        get_platform().barrier()

    return callback


def create_eval_callback(eval_name: str, loader: DataLoader, verbose=False):
    """This function returns a callback."""

    time_of_last_call = None

    def eval_callback(output_location, step, model, optimizer, logger):
        example_count = torch.tensor(0.0).to(get_platform().torch_device)
        total_loss = torch.tensor(0.0).to(get_platform().torch_device)
        total_correct = torch.tensor(0.0).to(get_platform().torch_device)

        def correct(labels, outputs):
            return torch.sum(torch.eq(labels, output.argmax(dim=1)))

        model.eval()

        with torch.no_grad():
            for examples, labels in loader:
                examples = examples.to(get_platform().torch_device)
                labels = labels.squeeze().to(get_platform().torch_device)
                output = model(examples)

                labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
                example_count += labels_size
                total_loss += model.loss_criterion(output, labels) * labels_size
                total_correct += correct(labels, output)

        # Share the information if distributed.
        if get_platform().is_distributed:
            torch.distributed.reduce(total_loss, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(total_correct, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(example_count, 0, op=torch.distributed.ReduceOp.SUM)

        total_loss = total_loss.cpu().item()
        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()

        if get_platform().is_primary_process:
            logger.add('{}_loss'.format(eval_name), step, total_loss / example_count)
            logger.add('{}_accuracy'.format(eval_name), step, total_correct / example_count)
            logger.add('{}_examples'.format(eval_name), step, example_count)

            if verbose:
                nonlocal time_of_last_call
                elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
                print('{}\tep {:03d}\tit {:03d}\tloss {:.3f}\tacc {:.2f}%\tex {:d}\ttime {:.2f}s'.format(
                    eval_name, step.ep, step.it, total_loss/example_count, 100 * total_correct/example_count,
                    int(example_count), elapsed))
                time_of_last_call = time.time()

    return eval_callback


# Callback frequencies. Each takes a callback as an argument and returns a new callback
# that runs only at the specified frequency.
def run_every_epoch(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.it != 0:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


def run_every_step(callback):
    return callback


def run_at_step(step1, callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step != step1:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


# The standard set of callbacks that should be used for a normal training run.
def standard_callbacks(training_hparams: hparams.TrainingHparams, train_set_loader: DataLoader,
                       test_set_loader: DataLoader, eval_on_train: bool = False, verbose: bool = True,
                       start_step: Step = None, evaluate_every_epoch: bool = True):
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    end = Step.from_str(training_hparams.training_steps, train_set_loader.iterations_per_epoch)
    test_eval_callback = create_eval_callback('test', test_set_loader, verbose=verbose)
    train_eval_callback = create_eval_callback('train', train_set_loader, verbose=verbose)

    # Basic checkpointing and state saving at the beginning and end.
    result = [
        run_at_step(start, save_model),
        run_at_step(end, save_model),
        run_at_step(end, save_logger),
        run_every_epoch(checkpointing.save_checkpoint_callback),
    ]

    # Test every epoch if requested.
    if evaluate_every_epoch: result = [run_every_epoch(test_eval_callback)] + result
    elif verbose: result.append(run_every_epoch(create_timekeeper_callback()))

    # Ensure that testing occurs at least at the beginning and end of training.
    if start.it != 0 or not evaluate_every_epoch: result = [run_at_step(start, test_eval_callback)] + result
    if end.it != 0 or not evaluate_every_epoch: result = [run_at_step(end, test_eval_callback)] + result

    # Do the same for the train set if requested.
    if eval_on_train:
        if evaluate_every_epoch: result = [run_every_epoch(train_eval_callback)] + result
        if start.it != 0 or not evaluate_every_epoch: result = [run_at_step(start, train_eval_callback)] + result
        if end.it != 0 or not evaluate_every_epoch: result = [run_at_step(end, train_eval_callback)] + result

    return result
