# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from foundations import paths
from foundations.step import Step
from platforms.platform import get_platform
from training.metric_logger import MetricLogger


def save_checkpoint_callback(output_location, step, model, optimizer, logger):
    if get_platform().is_primary_process:
        get_platform().save_model({
            'ep': step.ep,
            'it': step.it,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'logger': str(logger),
        }, paths.checkpoint(output_location))
    get_platform().barrier()


def restore_checkpoint(output_location, model, optimizer, iterations_per_epoch):
    checkpoint_location = paths.checkpoint(output_location)
    if not get_platform().exists(checkpoint_location):
        return None, None
    checkpoint = get_platform().load_model(checkpoint_location, map_location=torch.device('cpu'))

    # Handle DataParallel.
    module_in_name = get_platform().is_parallel
    if module_in_name and not all(k.startswith('module.') for k in checkpoint['model_state_dict']):
        checkpoint['model_state_dict'] = {'module.' + k: v for k, v in checkpoint['model_state_dict'].items()}
    elif all(k.startswith('module.') for k in checkpoint['model_state_dict']) and not module_in_name:
        checkpoint['model_state_dict'] = {k[len('module.'):]: v for k, v in checkpoint['model_state_dict'].items()}

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = Step.from_epoch(checkpoint['ep'], checkpoint['it'], iterations_per_epoch)
    logger = MetricLogger.create_from_string(checkpoint['logger'])

    return step, logger
