# OpenLTH: A Framework for Lottery Tickets and Beyond

### Welcome

This framework implements key experiments from recent work on the lottery ticket hypothesis and the science of deep learning:

* [_The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks._](https://openreview.net/forum?id=rJl-b3RcF7) Jonathan Frankle & Michael Carbin. ICLR 2019.
* [_Stabilizing the LTH/The LTH at Scale._](https://arxiv.org/abs/1903.01611) Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, & Michael Carbin. Arxiv.
* [_Linear Mode Connectivity and the LTH._](https://arxiv.org/abs/1912.05671) Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, & Michael Carbin. Arxiv.
* [_The Early Phase of Neural Network Training._](https://openreview.net/forum?id=Hkl1iRNFwS) Jonathan Frankle, David J. Schwab, and Ari S. Morcos. ICLR 2020.
* [_Training BatchNorm and Only BatchNorm._](https://arxiv.org/abs/2003.00152) Jonathan Frankle, David J. Schwab, & Ari S. Morcos. Arxiv

It was created by [Jonathan Frankle](http://www.jfrankle.com) during his time as a summer intern and student researcher at FAIR starting in June 2019. It is his current working research codebase for image classification experiments on the lottery ticket hypothesis.

### Citation

If you use this library in a research paper, please cite this repository.

### License

OpenLTH is licensed under the MIT license, as found in the LICENSE file.

### Contributing

We welcome your contributions! See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## Table of Contents

### [1 Overview](#overview)
### [2 Getting Started](#getting-started)
### [3 Internals](#internals)
### [4 Extending the Framework](#extending)
### [5 Acknowledgements](#acknowledgements)

## <a name=overview></a>1 Overview

### 1.1 Purpose

This framework is designed with four goals in mind:

1. To train standard neural networks for image classification tasks.
2. To run pruning and lottery ticket experiments.
3. To automatically manage experiments and results without any manual intervention.
4. To make it easy to add new datasets, models, and other modifications.

### 1.2 Why Another Framework?

This framework and its predecessors were developed in the course of conducting research on the lottery ticket hypothesis.
* **Pruning.** Those experiments involve pruning neural networks. Pruning is a first-class citizen of the framework.
* **Hyperparameter management.** Those experiments involve extensive hyperparameter sweeps. Default hyperparameters are easy to modify, and results are automatically indexed by the hyperparameters (so you never need to worry about naming experiments).
* **Extensibility.** Those experiments study an ever-growing range of models and datasets. Consistent abstractions for models and datasets make it easy to add new ones in a modular way.
* **Enabling new experiments.** Those experiments involve adding new hyperparameters. Each hyperparameter automatically surfaces on the command-line and integrates into experiment naming in a backwards-compatible way.
* **Re-training and ablation.** Those experiments rely on many post-hoc re-training and ablation experiments. Such experiments can be created by writing a single function, after which they are automatically available to be called from the command line and stored in a standard way.
* **Platform flexibility.** Those experiments have had to run on many different platforms and providers. The notion of a "platform" is a first-class abstraction, making it easy to adapt the framework to new settings or to multiple settings.

## <a name='getting-started'></a>2 Getting Started

### 2.1 Requirements

* Python 3.7 or greater (this framework extensively uses features from Python 3.7)
* [PyTorch 1.4 or greater](https://www.pytorch.org)
* TorchVision 0.5.0 or greater
* [NVIDIA Apex](https://anaconda.org/conda-forge/nvidia-apex) (optional, but required for 16-bit training)

### 2.2 Setup

1. Install the requirements.
2. Clone this repository.
3. Modify `platforms/local.py` so that it contains the paths where you want datasets and results to be stored. By default, they will be stored in `~/open_lth_data/` and `~/open_lth_datasets/`. To train with ImageNet, you will need to specify the path where ImageNet is stored.

### 2.3 The Command-Line Interface

All interactions with the framework occur through the command-line interface:

```
python open_lth.py
```

In response, you will see the following message.

```
==================================================================================
OpenLTH: A Framework for Research on Lottery Tickets and Beyond
----------------------------------------------------------------------------------
Choose a command to run:
    * open_lth.py train [...] => Train a model.
    * open_lth.py lottery [...] => Run a lottery ticket hypothesis experiment.
    * open_lth.py lottery_branch [...] => Run a lottery branch.
==================================================================================
```

This framework has three subcommands for its three experimental workflows: `train` (for training a network), `lottery` (for running a lottery ticket hypothesis experiment), and `lottery_branch` (for running an ablation on a lottery ticket experiment). To learn about adding more experimental workflows, see `foundations/README.md`.

### 2.4 Training a Network

To train a network, use the `train` subcommand. You will need to specify the model to be trained, the dataset on which to train it, and other standard hyperparameters (e.g., batch size, learning rate, training steps). There are two ways to do so:

* **Specify these hyperparameters by hand.** Each hyperparameter is controlled by a command-line argument. To see the complete list, run `python open_lth.py train --help`. Many hyperparameters are required (e.g., `--dataset_name`, `--model_name`, `--lr`). Others are optional (e.g., `--momentum`, `--random_labels_fraction`).
* **Use the defaults.** Each model comes with a set of defaults that achieve standard performance. You can specify the name of the model you wish to train using the `--default_hparams` argument and load the default hyperparameters for that model.  You can still override any default using the individual arguments for each hyperparameter.

In practice, you will almost always begin with a set of defaults and optionally modify individual hyperparameters as desired. To view the default hyperparameters for ResNet-20 on CIFAR-10, use the following command. (For a full list of available models, see 2.11.) Each of the hyperparameters from before will be updated with its default value.

```
python open_lth.py train --default_hparams=cifar_resnet_20 --help
```

To train with these default hyperparameters, use the following command (that is, leave off `--help`):

```
python open_lth.py train --default_hparams=cifar_resnet_20
```

The training process will then begin. The framework will print the required and non-default hyperparameters for the training run and the location where the resulting model will be stored.

```
==================================================================================
Training a Model (Replicate 1)
----------------------------------------------------------------------------------
Dataset Hyperparameters
    * dataset_name => cifar10
    * batch_size => 128
Model Hyperparameters
    * model_name => cifar_resnet_20
    * model_init => kaiming_normal
    * batchnorm_init => uniform
Training Hyperparameters
    * optimizer_name => sgd
    * lr => 0.1
    * training_steps => 160ep
    * momentum => 0.9
    * milestone_steps => 80ep,120ep
    * gamma => 0.1
    * weight_decay => 0.0001
Output Location: /home/jfrankle/open_lth_data/train_71bc92a970b64a76d7ab7681764b0021/replicate_1/main
==================================================================================
```

Before each epoch, it will print the test error and loss.

```
test    ep 000  it 000  loss 68.261     acc 10.00%       time 0.00s
```

To suppress these messages, use the `--quiet` command-line argument.

To override any default hyperparameter values, use the corresponding hyperparameter arguments. For example, to increase the batch size and learning rate and add 10 epochs of learning rate warmup:

```
python open_lth.py train --default_hparams=cifar_resnet_20 --batch_size=1024 --lr=0.8 --warmup_steps=10ep
```

### 2.5 Running a Lottery Ticket Experiment

A lottery ticket experiment involves repeatedly training the network to completion, pruning weights, _rewinding_ unpruned weights to their value at initialization, and retraining. For more details, see [_The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks_](https://arxiv.org/abs/1803.03635).

To run a lottery ticket experiment, use the `lottery` subcommand. You will need to specify all the same hyperparameters required for training. In addition, you will need to specify hyperparameters for pruning the network. To see the complete set of hyperparameters, run:

```
python open_lth.py lottery --help
```

For pruning, you will need to specify a value for the `--pruning_strategy` hyperparameter. By default, the framework includes only one pruning strategy: pruning the lowest magnitude weights globally in a sparse fashion (`sparse_global`). (For instructions on adding new pruning strategies, see Section 4.7.)

Once again, it is easiest to load the default hyperparameters for a model (which includes the pruning strategy and other pruning details) using the `--default_hparams` argument. In addition, you will need to specify the number of times that the network should be pruned, rewound, and retrained, known as the number of pruning _levels_. To do so, use the `--levels` flag. Level 0 is training the full network; specifying `--levels=3` would then prune, rewind, and retrain a further three times.

To run a lottery ticket experiment with the default hyperparameters for ResNet-20 on CIFAR-10 with three pruning levels, use the following command:

```
python open_lth.py lottery --default_hparams=cifar_resnet_20 --levels=3
```

### 2.6 Running a Lottery Ticket Experiment with Rewinding

In the original paper on the lottery ticket hypothesis, unpruned weights were always rewound to their values from initialization before the retraining phase. In recent work ([_Stabilizing the LTH/The LTH at Scale_](https://arxiv.org/abs/1903.01611), [_Linear Mode Connectivity and the LTH_](https://arxiv.org/abs/1912.05671) and [_The Early Phase of Neural Network Training_](https://arxiv.org/abs/2002.10365)), weights are typically rewound to their values from step `k` during training (rather than from initialization).

This framework incorporates that concept through a broader feature called _pretraining_. Optionally, the full network can be _pretrained_ for `k` steps, with the resulting weights used as the starting point for the lottery ticket procedure. Rewinding to step `k` and pretraining for `k` steps are functionally identical, but pretraining offers increased flexibility. For example, you can pretrain using a different training set, batch size, or loss function; this is precisely the experiment performed in Section 5 of [_The Early Phase of Neural Network Training_](https://arxiv.org/abs/2002.10365).

If you wish to use the same hyperparameters for pretraining and the training process itself (i.e., perform standard lottery ticket rewinding), you can set the argument `--rewinding_steps`. For example, to rewind to iteration 500 after pruning (or, equivalently, to pretrain for 500 iterations), use the following command:

```
python open_lth.py lottery --default_hparams=cifar_resnet_20 --levels=3 --rewinding_steps=500it
```

If you wish to have different behavior during the pre-training phase (e.g., to pre-train with self-supervised rotation or even on a different task), use the `--pretrain` argument.  After doing so, the `--help` interface will offer the full suite of pretraining hyperparameters, including learning rate, batch size, dataset, etc. By default, it will pretrain using the same hyperparameters as specified for standard training. You will noueed to set the `--pretrain_training_steps` argument to the number of steps for which you wish to pretrain. Note that the network will still only train for the number of steps specified in `--training_steps`. Any steps specified in `--pretrain_training_steps` will be subtracted from `--training_steps`. In addition, the main phase of training will start from step `--pretrain_training_steps`, including the learning rate and state of the dataset at that step.

### 2.7 Lottery Ticket Branches

Many of the experiments in the lottery ticket papers involve running ablations based on the main lottery ticket experiment. For example, training the same pruned network but with a different random initialization or training with the same random initialization but a random pruning mask. Typically, you run this ablation for the networks produced by the main lottery ticket experiment at each level of pruning. For example, if we run a lottery ticket experiment with three levels of pruning, we will run this ablation four times - once on the unpruned network and once on the network at each level of pruning. We refer to such experiments as _branches_, since they branch off of the main lottery ticket trunk.

To run a branch, you need to first run the corresponding lottery ticket trunk. Afterwards, use the `lottery_branch` subcommand. This subcommand must be followed by the name of the branch (a sub-sub command).

```
python open_lth.py lottery_branch
```

If you do not specify a branch name, the framework will list all available branches.

```
==================================================================================
OpenLTH: A Framework for Research on Lottery Tickets and Beyond
----------------------------------------------------------------------------------
Choose a branch to run:
    * open_lth.py lottery_branch randomly_prune [...] => Randomly prune the model.
    * open_lth.py lottery_branch randomly_reinitialize [...] => Randomly reinitialize the model.
    * open_lth.py lottery_branch retrain [...] => Retrain the model with different hyperparameters.
==================================================================================
```

Each branch takes the standard lottery ticket arguments (in order to specify the trunk off of which to branch) and several branch-specific arguments. To see these arguments, use the `--help` command. Many of these arguments have standard default values, so you may not need to specify some or all of them. Finally, each `lottery_branch` command runs for a range of pruning levels (for example, '0-2,5-8,12'), so you need to specify that level with the `--level` argument.

For example, to randomly reinitialize level 2 of the prior lottery ticket experiment with random seed `7`, use the following command:

```
python open_lth.py lottery_branch randomly_prune --default_hparams=cifar_resnet_20 --levels=0-20 --seed=7
```

### 2.8 Accessing Results

All experiments are automatically named according to their hyperparameters. Specifically, all required hyperparameters and all optional hyperparameters that are specified are combined in a canonical fashion and hashed. This hash is the name under which the experiment is stored. The results of a training run are then stored under:

```
<root>/train_<hash>/replicate_<replicate>/main
```

`<root>` is the data root directory stored in `platforms/local.py`; it defaults to `~/open_lth_data`.

The results themselves are stored in a file called `logger`, which only appears after training is complete. This file is a lightweight CSV where each line is one piece of telemetry data about the model. A line consists of three comma-separated values: the name of the kind of telemetry (e.g., `test-accuracy`), the iteration of training at which the telemetry data was collected (e.g., `391`), and the value itself. You can parse this file manually, or use `training/metric_logger.py`, which is used by the framework to read and write these files.

To get the name of the output location for a particular run, use the `--display_output_location` flag.

```
python open_lth.py train --default_hparams=cifar_resnet_20 --display_output_location
/home/jfrankle/open_lth_data/train_71bc92a970b64a76d7ab7681764b0021/replicate_1/main
```

### 2.9 Checkpointing and Re-running

Each experiment will automatically checkpoint after every epoch. If you re-launch a job, it will automatically pick up from there. If a job has already completed, it will not run again unless you manually delete the associated results.

If you wish to run multiple copies of an experiment (which is good scientific practice), use the `--replicate` argument. This optional argument specifies the replicate number of an experiment. For example, `--replicate=5` will store the experiment under

<pre>
/home/jfrankle/open_lth_data/train_71bc92a970b64a76d7ab7681764b0021/replicate_<b>5</b>/main
</pre>

rather than

<pre>
/home/jfrankle/open_lth_data/train_71bc92a970b64a76d7ab7681764b0021/replicate_<b>1</b>/main
</pre>

If no replicate is specified, `--replicate` will default to 1.

### 2.10 Modifying the Training Environment

To suppress the outputs, use the `--quiet` argument.

To specify the number of PyTorch worker threads used to load data, use the `--num_workers` argument. This value defaults to 0, although you will need to specify a value for training with ImageNet.

The framework will automatically use all available GPUs. To change this behavior, you will need to modify the number of visible GPUs using the `CUDA_VISIBLE_DEVICES` environment variable.

### 2.11 Available Models

The framework includes the following models. Each model family shares the same default hyperparameters.

|Model|Description|Name in Framework|Example|
|--|--|--|--|
|LeNet| A fully-connected MLP. | `mnist_mnist_N_M_L...` where N, M, and L are the number of neurons per layer. You can add as many layers as you like in this way. You do not need to include the output layer. | `mnist_lenet_300_100` |
|VGG for CIFAR-10| A convolutional network with max pooling in the style of VGG. | `cifar_vgg_D`, where `D` is the depth of the network (valid choices are 11, 13, 16 or 19). | `cifar_vgg_16` |
|ResNet for CIFAR-10| A residual network for CIFAR-10. This is a different family of architectures than those designed for ImageNet. | `cifar_resnet_D`, where `D` is the depth of the network. `D-2` must be divisible by 6 to be a valid ResNet | `cifar_resnet_20`, `cifar_resnet_110` |
|Wide ResNet for CIFAR-10| The ResNets for CIFAR-10 in which the width of the network can be varied. | `cifar_resnet_D_W`, where `D` is the depth of the network as above, and `W` is the number of convolutional filters in the first block of the network. If `W` is 16, then this network is equivalent to `cifar_resnet_D`; to double the width, set `W` to 32. | `cifar_resnet_20_128` |
|ResNet for ImageNet| A residual network for ImageNet. This is a different family of architectures than those designed for CIFAR-10, although they can be trained on CIFAR-10. | `imagenet_resnet_D`, where `D` is the depth of the network (valid choices are 18, 34, 101, 152, and 200) | `imagenet_resnet_50` |
|Wide ResNet for ImageNet| The ResNets for ImageNet in which the width of the network can be varied. | `imagenet_resnet_D_W`, where `D` is the depth of the network as above, and `W` is the number of convolutional filters in the first block of the network. If `W` is 64, then this network is equivalent to `imagenet_resnet_D`; to double the width, set `W` to 128. | `imagenet_resnet_50_128` |

### 2.12 ImageNet

This framework includes standard ResNet models for ImageNet and a standard data preprocessing for ImageNet. Using the default hyperparameters and 16-bit precision training, `imagenet_resnet_50` trains to 76.1% top-1 accuracy in 22 hours on four V100-16GB GPUs. To use ImageNet, you will have to take additional steps.

1. Prepare the ImageNet dataset.
    1. Create two folders, `train`, and `val`, each of which has one subfolder for each class containing the JPEG images of the examples in that class.
    2. Modify `imagenet_root()` in `platforms/local.py` to return this location.
2. If you wish to train with 16-bit precision, you will need to install the [NVIDIA Apex](https://anaconda.org/conda-forge/nvidia-apex) and add the `--apex_fp16` argument to the training command.

## <a name=internals></a>3 Internals

This framework is designed to be extensible, making it easy to add new datasets, models, initializers, optimizers, pruning strategies, hyperparameters, branches, workflows, and other customizations. This section discusses the internals. Section 4 is a how-to guide for extending the framework.

Note that this framework makes extensive use of Python [Data Classes](https://docs.python.org/3/library/dataclasses.html), a feature introduced in Python 3.7. You will need to understand this feature before you dive into the code. This framework also makes extensive use of object oriented subclassing with the help of the Python [ABC library](https://docs.python.org/3/library/abc.html).

### 3.1 Hyperparameters Abstraction

The lowest-level abstraction in the framework is an object that stores a bundle of hyperparameters. The abstract base class for all such bundles of hyperparameters is the `Hparams` Data Class, which can be found in `foundations/hparams.py`. This file also includes four subclasses of `Hparams` that are used extensively throughout the framework:
* `DatasetHparams` (which includes all hyperparameters necessary to specify a dataset, like its name and batch size)
* `ModelHparams` (which includes all hyperparameters necessary to specify a model, like its name and initializer)
* `TrainingHparams` (which includes all hyperparameters necessary to describe how to train a model, like the optimizer, learning rate, warmup, annealing, and number of training steps)
* `PruningHparams` (which is the base class for the hyperparameters required by each pruning strategy)

Each field of these dataclasses is the name of the hyperparameter. The type annotation is the type that the hyperparameter must have. If a default value is specified for the field, that is the default value for the hyperparameter; if no default is provided, then the hyperparameter is required and must be specified manually.

Each `Hparams` subclass must also set the `_name` and `_description` fields with default values that describe the nature of this bundle of hyperparameters. It may optionally include a string field `_hyperparameter` with a default string value that describes the hyperparameter and how it should be set. For example, in addition to the `lr` field, `TrainingHparams` has the `_lr` field that explains how the `lr` field should be set.

The `Hparams` subclass provides several behaviors to its subclasses. Most importantly, it has a static method `add_args` which takes as input a Python command-line `ArgumentParser` and adds each of the hyperparameters as a flag `--hyperparameter`. Since each hyperparameter has a name, type annotation, and possibly a default value and help text (the `_hyperparameter` field), it can be converted into a command-line argument automatically. This is how the per-hyperparameter command-line arguments are populated. This function optionally takes an instance of the class that overrides default values; this is how `--default_hparams` is implemented. Corresponding to the `add_args` static method is a `create_from_args` static method that creates an instance of the class from a Python `argparse.NameSpace` object that results from using the `ArgumentParser`.

Finally, the `Hparams` object has a `__str__` method that converts an instance into a string in a canonical way. During this conversion, any hyperparameters that are set to their default values are left off. This step is very important for ensuring that models are saved in a backwards compatible way as new hyperparameters are added.

### 3.2 Modules for Datasets, Models, Training, and Pruning

Running a lottery ticket experiment involves combining four largely independent components:
1. There must be a way to retrieve a dataset.
2. There must be a way to retrieve a model.
3. There must be a way to train a model on a dataset.
4. Three must be a way to prune a model.

This framework breaks these components into distinct modules that are as independent as possible. The common specification for these modules is the `Hparams` objects. To request a dataset from the dataset module, provide a `DatasetHparams` instance. To request a model from the models module, provide a `ModelHparams` instance. To train a model, provide a dataset, a model, and a `TrainingHparams` instance. To prune a model, provide the model and a `PruningHparams` object. The inner workings of these modules can be understood largely independently from each other, with a few final abstractions to glue everything together.

### 3.3 The Datasets Module

Each dataset consists of two abstractions:

1. A `Dataset` that stores the dataset, labels, and any data augmentation.
2. A `DataLoader` that loads the dataset for training or testing. It must keep track of the batch size, multithreaded infrastructure for data loading, and random shuffling.

A dataset must subclass the `Dataset` and `DataLoader` abstract base classes in `datasets/base.py`. Both of these classes subclass the corresponding PyTorch `Dataset` and `DataLoader` classes, although they have a richer API to facilitate functionality in other modules and to enable build-in transformations like subsampling, random labels, and blurring.

For simple datasets that can fit in memory, these base classes provide most of the necessary functionality, so the subclasses are small. In fact, MNIST (`datasets/mnist.py`) and CIFAR-10 (`datasets/cifar10.py`) use the base `DataLoader` without modification. In contrast, ImageNet (`datasets/imagenet.py`) replaces all functionality due to the specialized needs of loading such a large dataset efficiently.

The external interface of this module is contained in `datasets/registry.py`. The registry contains a list of all existing datasets in the framework (so that they can be discovered and loaded). Its most important function is `get()`, which takes as input a `DatasetHparams` instance and a boolean specifying whether to load the train or test set; it returns the `DataLoader` object corresponding to the `DatasetHparams` (i.e., with the right batch size and additional transformations). This module also contains a function for getting the number of `iterations_per_epoch()` and the `num_classes()` corresponding to a particular `DatasetHparams`, both of which are important for other modules.

### 3.4 The Models Module

Each model is created by subclassing the `Model` abstract base class in `models/base.py`. This base class is a valid PyTorch `Module` with several additional abstract methods that support other functionality throughout the framework. In particular, any subclass must have static methods to determine whether a string model name (e.g., `cifar_resnet_20`) is valid and to create a model object from a string name, a number of outputs, and an initializer.

It must also have instance properties that return the names of the tensors in the output layer (`output_layer_names`) and all tensors that are available for pruning (`prunable_layer_names` - by default just the kernels of convolutional and linear layers). These properties are used elsewhere in the framework for transfer learning, weight freezing, and pruning.

Finally, it must have a static method that returns the set of default hyperparameters for the corresponding model family (as `Hparams` objects); doing so makes it possible to load the default hyperparameters rather than specifying them one by one on the command line.

Otherwise, these models are identical to standard `Module`s in PyTorch.

The external interface of this module is contained in `models/registry.py`. Like `datasets/registry.py`, there is a list of all existing models in the framework so that they can be discovered and loaded. The registry similarly contains a `get()` function that, given a `ModelHparams` instance and the number of outputs, returns the corresponding `Model` as specified. In the course of creating a model, the `get()` function also loads the initializer and BatchNorm initializer specified in the `ModelHparams` instance. All initializers are functions stored in `models/initializers.py`, and no registration is necessary. All BatchNorm initializers are functions stored in `models/bn_initializers.py`, and no registration is necessary.

Finally, the registry has functions for getting the default hyperparameters for a model, loading a saved model from a path, and checking whether a path contains a saved model.

### 3.5 The Step Abstraction

In several places throughout the framework, it is necessary to keep track of a particular "step" of training. Depending on the particular framework, a step takes one of two forms: an iteration of training or an epoch number and an iteration offset into that epoch. In some places in this framework, it is easier to use one representation than the other. To make it easy to convert back and forth between these representations, all steps are stored as `Step` objects (`foundations/step.py`). A step object can be created from either representation, but it requires the number of iterations per epoch so that it can convert back and forth between these two representations.

### 3.6 The Training Module

The training module centers on a single function: the `train()` function in `training/train.py`. This function takes a `Model` and a `DataLoader` as arguments along with a `TrainingHparams` instance. It then trains the `Model` on the dataset provided by the `DataLoader` as specified in the training hyperparameters.

The `train()` function takes four other arguments. These include the `output_location` where all results should be stored, the optional `Step` object at which training should begin (the learning rate schedule and training set are advanced to this point), and the optional `Step` object at which training should end (if not the default value specified in the `TrainingHparams` instance).

Most importantly, it takes an argument called `callbacks`. This argument requires some explaining. A key design goal of the training module is to keep the main training loop in `train()` as clean as possible. This means that the loop should only contain standard setup, training, checkpointing behavior, and a `MetricLogger` to record telemetry. The loop should not be modified to add other behaviors, like saving the network state, running the test set, adding if-statements for new experiments, etc.

Instead, the behavior of the loop is modified by providing _callback_ functions (known as _hooks_ in other frameworks). These callbacks are called before every training step and after the final training step. They are provided with the current training step, the model, the optimizer, the output location, and the logger, and they can perform functions like saving the model state, running the test set, checkpointing, printing useful debugging information, etc. As new functionality is needed in the training loop, simply create new callbacks.

The file `training/standard_callbacks.py` contains a set of the most common callbacks you are likely to use, like evaluating the model on a `DataLoader` or saving the model state. It also contains a set of higher-order functions that modify a callback to run at a certain step or interval. Finally, it includes a set of standard callbacks for a training run:
* Save the network before and after training
* Run the test set every epoch
* Update the checkpoint every epoch
* Save the logger after training

The file `training/train.py` contains a function called `standard_train()` that takes a model, dataset and training hyperparameters, and an output location as inputs and trains the model using the standard callbacks and the main training loop. This function is used by the `train` and `lottery` subcommands.

To create optimizers and learning rate scheduler objects, `train()` calls the `get_optimizer()` and `get_lr_schedule()` functions in `training/optimizers.py`, which serve as small-scale registries for these objects.

### 3.7 The Pruning Module

The pruning module is in the `pruning/` directory. It contains a `Mask` abstraction, which keeps track of the binary mask `Tensor` for each prunable layer in a model. The module keeps track of different pruning strategy classes (subclasses of the abstract base class `Strategy` in `pruning/base.py`. Each pruning strategy has two members:
1. A static method `get_pruning_hparams()` that returns a subclass (_not_ an instance) of the `PruningHparams` class from `foundations/hparams.py`. Since different pruning methods may require different hyperparameters, each pruning method is permitted to specify its own `PruningHparams` object. This object is used to generate the command-line arguments for the pruning strategy specified by the `--pruning_strategy` argument.
2. A static method `prune` that takes a `PruningHparams` instance, a trained `Model`, and the current `Mask` and returns a new mask representing one further pruning step according to the hyperparameters.

The external interface of this module is contained in `pruning/registry.py`. Like the other registries, it has a `get()` function for getting a pruning class from a `PruningHparams` instance. It also has a `get_pruning_hparams` function for getting the `PruningHparams` subclass for a particular pruning strategy.

Finally, this module contains a `PrunedModel` class (in `pruning/pruned_model.py`). This class is a wrapper around a `Model` (from `models/base.py`) that applies a `Mask` object to prune weights. This class is used heavily by the lottery ticket and branch experiments to effectuate pruning.

### 3.8 Descriptors

These individual components (datasets, models, training, and pruning) come together to allow for training workflows. The framework currently has two training workflows: training a model normally (the `train` subcommand) and running a lottery ticket experiment (the `lottery` subcommand).

Each of these workflows requires a slightly different set of hyperparameters. Training a model requires `DatasetHparams`, `ModelHparams`, and `TrainingHparams` instances (but notably no `PruningHparams`, since no pruning occurs in this workflow). In contrast, a lottery ticket experiment also needs `PruningHparams` and, optionally, a separate set of `DatasetHparams` and `TrainingHparams` for pre-training.

In summary, each workflow needs a "bundle" of `Hparams` objects of different kinds. The framework represents this abstraction with a _descriptor_ object, which describes everything necessary to conduct the workflow (training a network or running a lottery ticket experiment). These objects descend from `foundations/desc.py`, which contains the abstract base class `Desc`. This class is a Python dataclass whose fields are `Hparams` objects. It requires subclasses to implement a `add_args` and `create_from_args` static methods that create the necessary command-line arguments like the similar methods in the `Hparams` base class; typically, these functions will simply call the corresponding ones in the constituent `Hparams` instances.

Importantly, the `Desc` base class contains the function that makes automatic experiment naming possible. It has a property called `hashname` that combines all `Hparams` objects in its fields into a single string in a canonical way and returns the MD5 hash. This hash later becomes the name under which each experiment is stored. It is therefore important to be careful when modifying `Hparams` or `Desc` objects, as doing so may break backwards compatibility with the hashes of pre-existing experiments.

The training and lottery workflows contain subclasses of `Desc` in `training/desc.py` and `lottery/desc.py`. Each of these subclasses contains the requisite fields and implements the required abstract methods. They also include other useful properties derived from their constituent hyperparameters for the convenience of higher-level abstractions.

### 3.9 Wiring Everything Together with Runners

A descriptor has everything necessary to specify how a particular network should be trained, but it is missing other meta-data necessary to fully describe a run. For example, a training run needs a replicate number, and a lottery run needs to know the number of pruning levels for which to run. This information is captured in a higher-level abstraction known as a `Runner`.

Each runner (subclasses of `Runner` in `foundations/runner.py`) has static `add_args` and `create_from_args` methods that interact with command-line arguments, calling the same methods on their requisite descriptors and adding other runner-level arguments. Once a `Runner` instance has been created, the `run()` method (which takes no arguments) initiates the run. This includes creating a model and making one or more calls to `train()` depending on the details of the runner. For example, the runner for the `train` subcommand (found in `training/runner.py`) performs a single training run; the runner for the `lottery` subcommand (found in `lottery/runner.py`) pretrains a network, trains it, prunes it, and then repeatedly re-trains and prunes it using the `PrunedModel` class and a pruning `Strategy`.

The runners are the highest level of abstraction, connecting directly to the command-line interface. Each runner must be registered in `cli/runner_registry.py`. The name under which it is added is the name of the subcommand used to access it on the command-line.

### 3.10 Branches

A lottery ticket training run with `N` pruning levels produces `N+1` networks: the original, unpruned network and the networks that result from each of the `N` pruning levels. In many lottery ticket experiments, some subsequent ablation is conducted on each of these networks. For example, they are randomly reinitialized and retrained to assess the role of the original initialization or they are randomly pruned to assess the role of the specific sparsity pattern found by pruning.

We refer to such experiments as _branches_, since they branch off of the main lottery ticket trunk. Branches are implemented in the `lottery/branch` directory. Each branch is implemented by writing a subclass of the `Branch` abstract base class in `lottery/branch/base.py`. This subclass must have three methods:
1. A static method `name` that returns the name of the branch as is used to refer to it on the command line.
2. A static method `description` that returns a description of the branch that is used on the command line.
3. An instance method `branch_function` that contains the body of the branch. This function receives all of the state of the superclass instance, including a `LotteryDesc` object describing the run being branched, the replicate being branched, and the level being branched. The `branch_function` method can have arguments that vary its behavior; these arguments are automatically converted into command-line arguments when this branch is called. The `branch_root` property of the superclass instance contains the name of the output directory that should be used for a call to this branch; it is generated based on the arguments with which the function is called.

The only remaining step is to register this branch in `lottery/branch/registry.py`.

The branch module is designed to make it as easy as possible to write a new branch, since this is a heavily-used path in lottery ticket research. All that it takes to write a branch is to write a `branch_function`; the base class takes care of connecting it to the command line and initializing its state automatically.

This magic relies on a little meta-programming to work. The base `Branch` class in `lottery/branch/base.py` uses the `__init_subclass__` function to modify any subclass that is created. It inspects the signature of the `branch_function` subclass, extracting the name, type annotation (which is required), and default value of each argument. Together, this information is enough to dynamically generate a `Hparams` class in which these arguments become hyperparameters.

Using that dynamically generated `Hparams` class, `__init_subclass__` dynamically generates a descriptor for the subclass using the `make_BranchDesc` function in `lottery/branch/desc.py`. This descriptor has one set of `Hparams` (those that were dynamically generated) and a `LotteryDesc` field containing the descriptor of the lottery ticket trunk that it is building on.

The `Branch` base class itself is also a `Runner`; it builds the descriptor and, when run, calls `branch_function` in the subclass, executing the branch.

To connect the branch to the command line, there is another runner (in `lottery/branch/runner.py`). When the `lottery_branch` subcommand is used, it requests the name of the desired branch as a sub-subcommand. Once that is provided, it consults the branch registry (`lottery/branch/registry.py`) and dispatches to the appropriate branch `Runner` to execute the job.

### 3.11 Platforms

It is typical to use the same codebase on many different infrastructures (such as a local machine, a cluster, and one or more cloud providers). Each of these infrastructures will have different locations where results and datasets will be stored and different ways of accessing filesystems. They may even need to call the runner's `run()` functions in a different fashion.

To make it easy to run this framework on multiple infrastructures, it includes a `Platform` abstraction. Each `Platform` class describes where to find resources (like datasets), where to store results, what hardware is available (if there are GPUs and how many if so) and how to run a job on the platform. Arguments may be required to create a `Platform` instance, for example the number of worker threads to use.

To enable this behavior, each `Platform` object is a dataclass that descends from `Hparams`; this makes it possible for its fields to be converted into command-line arguments and for an instance to be created from these arguments. The abstract base `Platform` class that all others subclass (found in `platforms/base.py`) contains a field for the number of worker threads to use for data loading.
It also has abstract properties that specify where data should be found and results stored; these must be implemented by each subclass.
Finally, it has a series of static methods that mediate access to the filesystem; by default, these are set to use the standard Python commands for the local filesystem, but it may be important to override them on certain infrastructures.

Finally, it has a method called `run_job()` that receives a function `f` as an argument, performs any pre-job setup, and calls the function. Most importantly, this function _installs_ the `Platform` instance as the global platform for the entire codebase. In practice, this entails modifying the global variable `_PLATFORM` in `platforms/platform.py`. Throughout the codebase, modules look to this global variable (accessed through the `get_platform()` function in `platforms/platform`) to determine where data is stored, the hardware on which to run a job, etc. It was cleaner to make the current platform instance a global rather than to carry it along through every function call in the codebase.

The included `local` platform will automatically use all GPUs available using PyTorch `DataParallel`. If you choose to do distributed training, the `base` platform includes primitives for distributed training like `rank`,  `world_size`, `is_primary_process`, and `barrier`; the codebase calls all of these functions in the proper places so that it is forward-compatible with distributed training should you choose to use it.

All platform subclasses must be registered in `platforms/registry.py`, which makes them available for use at the command line using the `--platform` argument. By default the `local` platform (which runs on the local machine) is used.

### 3.12 Testing

This codebase contains extensive unit tests for the low-level modules and the lottery ticket hypothesis runner pipeline. To execute these tests, run the following command:

```
python -m unittest discover
```

The unit tests include a regression test for the directory names generated by the framework to ensure that new hyperparameters have not inadvertently changed existing names. Note that the unit tests do not directly test the command-line interface, the train `Runner`, and the branch infrastructure.

## <a name=extending></a>4 Extending the Framework

Please read Section 3 before trying to extend the framework. Careless changes can have unexpected consequences, such as breaking backwards compatibility and making it impossible for the framework to access your existing models.

### 4.1 Adding a New Dataset

Create a new file in the `datasets` directory that subclasses the abstract base classes `Dataset` and `DataLoader` in `datasets/base.py` with classes that are also called `Dataset` and `DataLoader`. Modify `datasets/registry.py` to import this module and add the module (_not_ the classes in the module) to the dictionary of `registered_datasets` with the name that you wish for it to be called. For small datasets that fit in memory (e.g., SVHN), use `datasets/cifar10.py` as a template and take advantage of functionality built into the base classes. For larger datasets (e.g., Places), use `datasets/imagenet.py` as a template; you may need to throw away functionality in the base classes.

### 4.2 Adding a New Model

Create a new file in the `models` directory that subclasses the abstract base class `Model` in `models/base.py`. Modify `models/registry.py` to import this module and add the class (_not_ the module containing the class) to the list of `registered_models`. As a template, use `models/cifar_resnet.py`.

### 4.3 Adding a New Initializer

Add the new initializer function to `models/initializers.py` under the name that you want it to be called. To add a new BatchNorm initializer, do the same in `models/bn_initializers.py`. No registration is necessary in either case.

### 4.4 Adding a New Optimizer

Modify the if-statement in the `get_optimizer` function of `training/optimizers.py` to create the new optimizer when the appropriate hyperparameters are specified.

### 4.5 Adding a New Hyperparameter

Modify the appropriate set of hyperparameters in `foundations/hparams.py` to include the desired hyperparameter. **The hyperparameter must have a default value, and this default value must eliminate the effect of the hyperparameter.** The goal is to ensure that adding this hyperparameter is backwards compatible. This default value should ensure that all preexisting models would train in the same way if this hyperparameter had been present and set to its default value.

If the new hyperparameter doesn't have a default value, then it will change the way results directory names are computed for all preexisting models, making it impossible for the framework to find them. If the default value is not a no-op, then all preexisting models (where were trained under the implicit assumption that this hyperparameter was set to its default value) will no longer be valid.

The unit tests include a regression test for the directory names generated by the framework to ensure that new hyperparameters have not inadvertently changed existing names. Be sure to run the unit tests after adding a new hyperparameter.

### 4.6 Modifying the Training Loop

Where possible, try to modify the training loop by creating a new kind of optimizer, a new kind of loss function, or a new callback. New callbacks can be added to `standard_train()` in `training/train.py`, gated by a new hyperparameter. The training loop is designed to be as clean and pared down as possible and to use callbacks and the other objects to abstract away the complicated parts, so try to avoid modifying the loop if at all possible. If you need to access the gradients, consider adding a second set of `post_gradient_callbacks` that are called after the gradients are computed but before the optimizer steps. This would be a new argument for `train()` and possibly `standard_train()` in `training/train.py`.

### 4.7 Adding a New Pruning Strategy

Create a new file in the `pruning` directory that subclasses the abstract base class `Strategy` in `pruning/base.py`. The new pruning strategy needs a static method that returns the hyperparameters it requires (recall that each pruning method can have a different set of hyperparameters). Modify `pruning/registry.py` to import this module and add the class (_not_ the module containing the class) to the dictionary of `registered_strategies` under the key that you want to use to describe this strategy going forward.

### 4.8 Adding a New Workflow

1. Create a new directory to store the workflow.
2. Create a file with a descriptor data class that subclasses from `Desc` in `foundations/desc.py`; it should have fields for any `Hparams` objects necessary to describe the workflow. It should implement the `add_args`, `create_from_args`, and `name_prefix` static methods as necessary for the desired behavior.
3. Create a file with a runner class that subclasses from `Runner` in `foundations/runner.py`. Create a constructor or make the runner a data class. Implement the `add_args` and `create_from_args` static methods to interact with the command line. Implement the `description` static method to describe the runner. Implement the `display_output_location` instance method to respond to the `--display_output_location` command-line argument. Finally, create the `run` instance method with the logic for performing any training necessary for the workflow.
4. Register the runner in `cli/runner_registry.py`.

### 4.9 Adding a New Branch

Create a new file in the `lottery/branch` directory that subclasses the abstract base class `Branch` in `lottery/branch/base.py`. This subclass needs three functions:

1. An instance method called `branch_function`. This function executes the branch, including any training that is required. The first argument must be `self` (the instance). Give the function any additional arguments that you want to appear as command-line arguments when this branch is run. These arguments must have type annotations, and they may have default values. The arguments may only be of Python value types (`int`, `float`, `bool`, `str`) or of a subclass of `Hparams`.
2. A static method called `name` that returns the name of the branch (for use on the command line).
3. A static method called `description` that returns a description of the branch (for use on the command line).

Finally, register this branch in `lottery/branch/registry.py`

### 4.10 Adding a New Platform

Subclass the `Platform` class (from `platforms/base.py`) in a new file in the `platforms` directory. Be sure to make it a dataclass. Add any additional fields and, optionally, help strings for these fields (named `_f` for a field `f`). Implement all the required abstract properties (`root`, `dataset_root`, and `imagenet_root` if ImageNet is available). Finally, override `run_job()` if different behavior is needed for the platform; be sure to ensure that the modified `run_job()` method still installs the platform instance before calling the job function `f`.


## <a name=acknowledgements></a>5 Acknowledgements

Thank you to Ari Morcos and David Schwab for supporting the development of this framework and the research that we conducted with it. Thank you to FAIR for allowing me to open-source this framework. Thank you to David Bieber for teaching me how to do software engineering around deep learning and for the many ideas I borrowed from [Python Fire](https://github.com/google/python-fire).
