from datasets.base import DataLoader
from foundations import hparams
from foundations.step import Step


def post_gradient_callbacks(training_hparams: hparams.TrainingHparams, train_set_loader: DataLoader,
                            test_set_loader: DataLoader, eval_on_train: bool = False, verbose: bool = True,
                            start_step: Step = None, evaluate_every_epoch: bool = True):
    result = []
    return result
