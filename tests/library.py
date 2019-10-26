#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf

import src.logictensornetworks.logictensornetworks as ltn

tf_variable_type = type(tf.Variable(1))


class TestEqual(unittest.TestCase):

    def test_constant_creation(self):
        const = ltn.constant(min_value=[0.] * 20, max_value=[1.] * 20)
        self.assertIsInstance(const, tf_variable_type)


if __name__ == "__main__":
    unittest.main()
