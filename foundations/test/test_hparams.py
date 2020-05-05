# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass

from foundations import hparams
from testing import test_case


class TestHparams(test_case.TestCase):
    class ArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError(message)

    def test_name_and_description(self):
        # _name and _description missing.
        with self.assertRaises(ValueError):
            @dataclass
            class Child(hparams.Hparams):
                x: int

            Child(2)

        # _name missing missing.
        with self.assertRaises(ValueError):
            @dataclass
            class Child(hparams.Hparams):
                x: int
                _description: str = 'Hello!'

            Child(2)

        # _description missing missing.
        with self.assertRaises(ValueError):
            @dataclass
            class Child(hparams.Hparams):
                x: int
                _name: str = 'Hello!'

            Child(2)

        # Both present
        @dataclass
        class Child(hparams.Hparams):
            x: int
            _name: str = 'Hello!'
            _description: str = 'World'

        Child(2)

    def test_canonical_name(self):
        @dataclass
        class Child(hparams.Hparams):
            x: int
            z: str = '2'
            y: str = '3'
            w: float = None
            _name: str = 'Hello!'
            _description: str = 'World!'
            _x = 'x help'
            _y = 'y help'

        self.assertEqual(str(Child(1)), 'Hparams(x=1)')
        self.assertEqual(str(Child(1, '5')), 'Hparams(x=1, z=\'5\')')
        self.assertEqual(str(Child(1, '2')), 'Hparams(x=1)')
        self.assertEqual(str(Child(1, '4', '6')), "Hparams(x=1, y='6', z='4')")
        self.assertEqual(str(Child(1, w=2.3)), 'Hparams(w=2.3, x=1)')

    def test_add_args(self):
        @dataclass
        class Child(hparams.Hparams):
            x: int
            z: str = '2'
            y: bool = False
            w: float = None
            _name: str = 'Hello!'
            _description: str = 'World!'
            _x = 'x help'
            _y = 'y help'

        class ArgumentParser(argparse.ArgumentParser):
            def error(self, message):
                raise ValueError(message)

        parser = ArgumentParser()
        Child.add_args(parser)

        # Test without providing 'x'
        with self.assertRaises(ValueError):
            parser.parse_args([])

        # Test with providing x.
        args = parser.parse_args(['--x=5'])
        self.assertTrue(hasattr(args, 'x'))
        self.assertTrue(hasattr(args, 'z'))
        self.assertTrue(hasattr(args, 'y'))
        self.assertTrue(hasattr(args, 'w'))
        self.assertFalse(hasattr(args, '_name'))
        self.assertFalse(hasattr(args, '_description'))
        self.assertFalse(hasattr(args, '_x'))
        self.assertFalse(hasattr(args, '_y'))

        self.assertEqual(args.x, 5)
        self.assertEqual(args.y, False)
        self.assertEqual(args.z, '2')
        self.assertEqual(args.w, None)

        c = Child.create_from_args(args)
        self.assertEqual(c.x, 5)
        self.assertEqual(c.y, False)
        self.assertEqual(c.z, '2')
        self.assertEqual(c.w, None)

        with self.assertRaises(ValueError):
            Child.create_from_args(args, prefix='jf')

        # Test with providing x and y.
        args = parser.parse_args(['--x=5', '--y'])
        self.assertEqual(args.x, 5)
        self.assertEqual(args.y, True)
        self.assertEqual(args.z, '2')
        self.assertEqual(args.w, None)

        c = Child.create_from_args(args)
        self.assertEqual(c.x, 5)
        self.assertEqual(c.y, True)
        self.assertEqual(c.z, '2')
        self.assertEqual(c.w, None)

        # Test with providing x and z.
        args = parser.parse_args(['--x=5', '--z=20ep'])
        self.assertEqual(args.x, 5)
        self.assertEqual(args.y, False)
        self.assertEqual(args.z, '20ep')
        self.assertEqual(args.w, None)

        c = Child.create_from_args(args)
        self.assertEqual(c.x, 5)
        self.assertEqual(c.y, False)
        self.assertEqual(c.z, '20ep')
        self.assertEqual(c.w, None)

        # Test with providing x, z, and w.
        args = parser.parse_args(['--x=5', '--w=8.5', '--z=20ep'])
        self.assertEqual(args.x, 5)
        self.assertEqual(args.y, False)
        self.assertEqual(args.z, '20ep')
        self.assertEqual(args.w, 8.5)

        c = Child.create_from_args(args)
        self.assertEqual(c.x, 5)
        self.assertEqual(c.y, False)
        self.assertEqual(c.z, '20ep')
        self.assertEqual(c.w, 8.5)

    def test_add_args_with_defaults_and_prefix(self):
        @dataclass
        class Child(hparams.Hparams):
            x: int
            z: str = '2'
            y: bool = False
            w: float = None
            _name: str = 'Hello!'
            _description: str = 'World!'
            _x = 'x help'
            _y = 'y help'

        defaults = Child(x=7, z='hello', w=2.3)

        class ArgumentParser(argparse.ArgumentParser):
            def error(self, message):
                raise ValueError(message)

        parser = ArgumentParser()
        Child.add_args(parser, defaults=defaults, prefix='jf')

        # Test without providing 'x'
        args = parser.parse_args([])
        self.assertTrue(hasattr(args, 'jf_x'))
        self.assertTrue(hasattr(args, 'jf_z'))
        self.assertTrue(hasattr(args, 'jf_y'))
        self.assertTrue(hasattr(args, 'jf_w'))
        self.assertFalse(hasattr(args, 'x'))
        self.assertFalse(hasattr(args, 'z'))
        self.assertFalse(hasattr(args, 'y'))
        self.assertFalse(hasattr(args, 'w'))
        self.assertFalse(hasattr(args, '_name'))
        self.assertFalse(hasattr(args, '_description'))
        self.assertFalse(hasattr(args, '_x'))
        self.assertFalse(hasattr(args, '_y'))

        self.assertEqual(args.jf_x, 7)
        self.assertEqual(args.jf_y, False)
        self.assertEqual(args.jf_z, 'hello')
        self.assertEqual(args.jf_w, 2.3)

        c = Child.create_from_args(args, prefix='jf')
        self.assertEqual(c.x, 7)
        self.assertEqual(c.y, False)
        self.assertEqual(c.z, 'hello')
        self.assertEqual(c.w, 2.3)

        # Ensure that the usual arguments aren't accepted
        with self.assertRaises(ValueError):
            parser.parse_args(['--x=5'])

        # Fails without specifying prefix.
        with self.assertRaises(ValueError):
            Child.create_from_args(parser.parse_args('--jf_x=5'))

        # Test with providing x.
        args = parser.parse_args(['--jf_x=5'])
        self.assertEqual(args.jf_x, 5)
        self.assertEqual(args.jf_y, False)
        self.assertEqual(args.jf_z, 'hello')
        self.assertEqual(args.jf_w, 2.3)

        c = Child.create_from_args(args, prefix='jf')
        self.assertEqual(c.x, 5)
        self.assertEqual(c.y, False)
        self.assertEqual(c.z, 'hello')
        self.assertEqual(c.w, 2.3)

        # Test with providing x and y.
        args = parser.parse_args(['--jf_x=5', '--jf_y'])
        self.assertEqual(args.jf_x, 5)
        self.assertEqual(args.jf_y, True)
        self.assertEqual(args.jf_z, 'hello')
        self.assertEqual(args.jf_w, 2.3)

        c = Child.create_from_args(args, prefix='jf')
        self.assertEqual(c.x, 5)
        self.assertEqual(c.y, True)
        self.assertEqual(c.z, 'hello')
        self.assertEqual(c.w, 2.3)

        # Test with providing x and z.
        args = parser.parse_args(['--jf_x=5', '--jf_z=20ep'])
        self.assertEqual(args.jf_x, 5)
        self.assertEqual(args.jf_y, False)
        self.assertEqual(args.jf_z, '20ep')
        self.assertEqual(args.jf_w, 2.3)

        c = Child.create_from_args(args, prefix='jf')
        self.assertEqual(c.x, 5)
        self.assertEqual(c.y, False)
        self.assertEqual(c.z, '20ep')
        self.assertEqual(c.w, 2.3)

    def test_nested_hparams(self):
        @dataclass
        class Nested(hparams.Hparams):
            a: int
            b: str = 'bb'
            _name: str = 'NestedName'
            _description: str = 'NestedDesc'

        @dataclass
        class Child(hparams.Hparams):
            x: int
            y: Nested
            z: str = '2'
            _name: str = 'ChildName'
            _description: str = 'ChildDesc'

        parser = TestHparams.ArgumentParser()
        Child.add_args(parser)

        with self.assertRaises(ValueError):
            parser.parse_args()

        with self.assertRaises(ValueError):
            parser.parse_args(['--x=5'])

        with self.assertRaises(ValueError):
            parser.parse_args(['--x=5', '--y=4'])

        with self.assertRaises(ValueError):
            parser.parse_args(['--x=5', '--a=2', '--y=3'])

        with self.assertRaises(ValueError):
            parser.parse_args(['--x=5', '--y_a=2', '--y=3'])

        # Test basic behavior.
        args = parser.parse_args(['--x=5', '--y_a=4'])
        self.assertEqual(args.x, 5)
        self.assertEqual(args.z, '2')
        self.assertEqual(args.y_a, 4)
        self.assertEqual(args.y_b, 'bb')
        self.assertFalse(hasattr(args, 'y'))
        self.assertFalse(hasattr(args, 'a'))
        self.assertFalse(hasattr(args, 'b'))

        c = Child.create_from_args(args)
        self.assertEqual(c.x, 5)
        self.assertEqual(c.z, '2')
        self.assertIsInstance(c.y, Nested)
        self.assertEqual(c.y.a, 4)
        self.assertEqual(c.y.b, 'bb')


test_case.main()
