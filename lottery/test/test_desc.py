# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from foundations import hparams
from lottery.desc import LotteryDesc
from pruning.sparse_global import Strategy
from testing import test_case


class TestDesc(test_case.TestCase):
    def test_hashes_regression(self):
        """Test that the hashes are fixed and repeatable.

        None of these hashes should change through any modification you make to the code.
        You can avoid this by ensuring you don't change existing hyperparameters and only
        add new hyperparameters that have default values.
        If they do, you will no longer be able to access your old models.
        """
        desc = LotteryDesc(
            dataset_hparams=hparams.DatasetHparams('cifar10', 128),
            model_hparams=hparams.ModelHparams('cifar_resnet_20', 'kaiming_normal', 'uniform'),
            training_hparams=hparams.TrainingHparams('sgd', 0.1, '160ep'),
            pruning_hparams=Strategy.get_pruning_hparams()('sparse_global')
        )
        self.assertEqual(desc.hashname, 'lottery_da8fd50859ba6d59aceca9d50ebcbf7e')

        with self.subTest():
            desc.training_hparams.momentum = 0.9
            self.assertEqual(desc.hashname, 'lottery_028eb999ecd1980cd012589829c945a3')

        with self.subTest():
            desc.training_hparams.milestone_steps = '80ep,120ep'
            desc.training_hparams.gamma = 0.1
            self.assertEqual(desc.hashname, 'lottery_e696cbf42d8758b8afdf2a16fad1de15')

        with self.subTest():
            desc.training_hparams.weight_decay = 1e-4
            self.assertEqual(desc.hashname, 'lottery_93bc65d66dfa64ffaf2a0ab105433a2c')

        with self.subTest():
            desc.training_hparams.warmup_steps = '20ep'
            self.assertEqual(desc.hashname, 'lottery_4e7b9ee929e8b1c911c5295233e6828f')

        with self.subTest():
            desc.training_hparams.data_order_seed = 0
            self.assertEqual(desc.hashname, 'lottery_d51482c0d378de4cc71b87b38df2ea84')

        with self.subTest():
            desc.dataset_hparams.do_not_augment = True
            self.assertEqual(desc.hashname, 'lottery_231b1efe748045875f738d860f4cb547')

        with self.subTest():
            desc.dataset_hparams.transformation_seed = 0
            self.assertEqual(desc.hashname, 'lottery_4dfd57a481be9a2d840f7ad5d1e6f5f0')

        with self.subTest():
            desc.dataset_hparams.subsample_fraction = 0.5
            self.assertEqual(desc.hashname, 'lottery_59ea6f2fab91a9515ae4bccd5de70878')

        with self.subTest():
            desc.dataset_hparams.random_labels_fraction = 0.7
            self.assertEqual(desc.hashname, 'lottery_8b59e5a4d5d72575f1fba67b476899fc')

        with self.subTest():
            desc.dataset_hparams.unsupervised_labels = 'rotation'
            self.assertEqual(desc.hashname, 'lottery_81f340e038ec29ffa9f858d9a8762211')

        with self.subTest():
            desc.dataset_hparams.blur_factor = 4
            self.assertEqual(desc.hashname, 'lottery_4e78e2719ef5c16ba3e0444bc10dfb08')

        with self.subTest():
            desc.model_hparams.batchnorm_frozen = True
            self.assertEqual(desc.hashname, 'lottery_8db76b3c3a08c4a1643f066768ff4e56')

        with self.subTest():
            desc.model_hparams.batchnorm_frozen = False
            desc.model_hparams.others_frozen = True
            self.assertEqual(desc.hashname, 'lottery_3a0f8b86c0813802537aea2ebe723051')

        with self.subTest():
            desc.model_hparams.others_frozen = False
            desc.pruning_hparams.pruning_layers_to_ignore = 'fc.weight'
            self.assertEqual(desc.hashname, 'lottery_d74aca8d02109ec0816739c2f7057433')


test_case.main()
