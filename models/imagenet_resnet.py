# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import torch
import torchvision

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class ResNet(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=1000, width=64):
        """To make it possible to vary the width, we need to override the constructor of the torchvision resnet."""

        torch.nn.Module.__init__(self)  # Skip the parent constructor. This replaces it.
        self._norm_layer = torch.nn.BatchNorm2d
        self.inplanes = width
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # The initial convolutional layer.
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The subsequent blocks.
        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width*2, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, width*4, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, width*8, layers[3], stride=2, dilate=False)

        # The last layers.
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(width*8*block.expansion, num_classes)

        # Default init.
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


class Model(base.Model):
    """A residual neural network as originally designed for ImageNet."""

    def __init__(self, model_fn, initializer, outputs=None):
        super(Model, self).__init__()

        self.model = model_fn(num_classes=outputs or 1000)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.apply(initializer)

    def forward(self, x):
        return self.model(x)

    @property
    def output_layer_names(self):
        return ['model.fc.weight', 'model.fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('imagenet_resnet_') and
                4 >= len(model_name.split('_')) >= 3 and
                model_name.split('_')[2].isdigit() and
                int(model_name.split('_')[2]) in [18, 34, 50, 101, 152, 200])

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=1000):
        """Name: imagenet_resnet_D[_W].

        D is the model depth (e.g., 50 for ResNet-50). W is the model width - the number of filters in the first
        residual layers. By default, this number is 64."""

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        num = int(model_name.split('_')[2])
        if num == 18: model_fn = partial(ResNet, torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
        elif num == 34: model_fn = partial(ResNet, torchvision.models.resnet.BasicBlock, [3, 4, 6, 3])
        elif num == 50: model_fn = partial(ResNet, torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        elif num == 101: model_fn = partial(ResNet, torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
        elif num == 152: model_fn = partial(ResNet, torchvision.models.resnet.Bottleneck, [3, 8, 36, 3])
        elif num == 200: model_fn = partial(ResNet, torchvision.models.resnet.Bottleneck, [3, 24, 36, 3])
        elif num == 269: model_fn = partial(ResNet, torchvision.moedls.resnet.Bottleneck, [3, 30, 48, 8])

        if len(model_name.split('_')) == 4:
            width = int(model_name.split('_')[3])
            model_fn = partial(model_fn, width=width)

        return Model(model_fn, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        """These hyperparameters will reach 76.1% top-1 accuracy on ImageNet.

        To get these results with a smaller batch size, scale the batch size linearly.
        That is, batch size 512 -> lr 0.2, 256 -> 0.1, etc.
        """

        model_hparams = hparams.ModelHparams(
            model_name='imagenet_resnet_50',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='imagenet',
            batch_size=1024,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='30ep,60ep,80ep',
            lr=0.4,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='90ep',
            warmup_steps='5ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
