# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from foundations import paths
from foundations.hparams import ModelHparams
from foundations.step import Step
from models import cifar_resnet, cifar_vgg, mnist_lenet, imagenet_resnet
from models import bn_initializers, initializers
from platforms.platform import get_platform

registered_models = [mnist_lenet.Model, cifar_resnet.Model, cifar_vgg.Model, imagenet_resnet.Model]


def get(model_hparams: ModelHparams, outputs=None):
    """Get the model for the corresponding hyperparameters."""

    # Select the initializer.
    if hasattr(initializers, model_hparams.model_init):
        initializer = getattr(initializers, model_hparams.model_init)
    else:
        raise ValueError('No initializer: {}'.format(model_hparams.model_init))

    # Select the BatchNorm initializer.
    if hasattr(bn_initializers, model_hparams.batchnorm_init):
        bn_initializer = getattr(bn_initializers, model_hparams.batchnorm_init)
    else:
        raise ValueError('No batchnorm initializer: {}'.format(model_hparams.batchnorm_init))

    # Create the overall initializer function.
    def init_fn(w): bn_initializer(initializer(w))

    # Select the model.
    model = None
    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_hparams.model_name):
            model = registered_model.get_model_from_name(model_hparams.model_name, init_fn, outputs)
            break

    if model is None:
        raise ValueError('No such model: {}'.format(model_hparams.model_name))

    # Freeze various subsets of the network.
    bn_names = []
    for k, v in model.named_modules():
        if isinstance(v, torch.nn.BatchNorm2d):
            bn_names += [k + '.weight', k + '.bias']

    if model_hparams.others_frozen_exceptions:
        others_exception_names = model_hparams.others_frozen_exceptions.split(',')
        for name in others_exception_names:
            if name not in model.state_dict():
                raise ValueError(f'Invalid name to except: {name}')
    else:
        others_exception_names = []

    for k, v in model.named_parameters():
        if k in bn_names and model_hparams.batchnorm_frozen:
            v.requires_grad = False
        elif k in model.output_layer_names and model_hparams.output_frozen:
            v.requires_grad = False
        elif k not in bn_names and k not in model.output_layer_names and model_hparams.others_frozen:
            if k in others_exception_names: continue
            v.requires_grad = False

    return model


def load(save_location: str, save_step: Step, model_hparams: ModelHparams, outputs=None):
    state_dict = get_platform().load_model(paths.model(save_location, save_step))
    model = get(model_hparams, outputs)
    model.load_state_dict(state_dict)
    return model


def exists(save_location, save_step):
    return get_platform().exists(paths.model(save_location, save_step))


def get_default_hparams(model_name):
    """Get the default hyperparameters for a particular model."""

    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_name):
            params = registered_model.default_hparams()
            params.model_hparams.model_name = model_name
            return params

    raise ValueError('No such model: {}'.format(model_name))
