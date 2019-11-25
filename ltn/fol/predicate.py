"""
:Author: thadumi
:Date: 25/11/19
:Version: 0.0.2
"""

import logging
from typing import Optional, Callable

import tensorflow as tf

import ltn.backend.fol_status as FOL
from logic import LogicalComputation
from ltn.backend.ltn_utils import cross_args


class LogicalPredicate(LogicalComputation):
    def __init__(self,
                 name=None,
                 input_doms=None,
                 output_doms=None,
                 input_term=None):
        super(LogicalPredicate, self).__init__(input_doms, output_doms, [])
        self.predicate = name
        self.input_terms = input_term

    def __str__(self):
        return self.predicate + '(' + ', '.join([str(i) for i in self.input_terms]) + ')'


class Predicate(object):
    def __init__(self, **kwargs):
        print(kwargs)
        self.name = kwargs['name']
        self.predicate_definition = kwargs['predicate_definition']
        self.number_of_arguments = kwargs['number_of_arguments']
        self.argument_size = kwargs['argument_size']

    def __call__(self, *args: LogicalComputation, **kwargs) -> LogicalPredicate:
        # Friends(c,v) -> LogicalPredicate which knows the arguments and what should be return
        input_doms = [arg.doms for arg in args]
        tensor_cross_args, output_doms = cross_args(input_doms)

        return LogicalPredicate(self.name,
                                input_doms, output_doms,
                                [*args])


def predicate(name: str,
              number_of_arguments: int,
              argument_size: int,
              predicate_definition: Optional[Callable[[tf.Tensor, ], tf.Tensor]] = None) -> Predicate:
    """
    TODO(thadumi) doc for predicate
    :param name:
    :param number_of_arguments:
    :param argument_size:
    :param predicate_definition:
    :return:
    """

    if FOL.predicate_already_defined(name):
        msg = '[predicate] There is already a predicate having the name `{}`'.format(name)
        logging.error(msg)

        raise ValueError(msg)

    if number_of_arguments <= 0:
        raise ValueError

    if argument_size <= 0:
        raise ValueError

    number_of_features = number_of_arguments * argument_size

    if predicate_definition is None:
        predicate_definition = DefaultPredicateModel(number_of_features)

    config = {'name': name,
              'predicate_definition': predicate_definition,
              'number_of_arguments': number_of_arguments,
              'argument_size': argument_size}

    p = Predicate(**config)
    FOL.track_predicate(name, p)
    return p


class DefaultPredicateModel(object):
    def __init__(self, number_of_features):
        self.w = FOL.variable(
            tf.random.normal([4, number_of_features + 1, number_of_features + 1],
                             mean=0,
                             stddev=1)
        )

        self.u = FOL.variable(tf.ones([4, 1]))

    @tf.function
    def __call__(self, *args) -> tf.Tensor:
        W = tf.linalg.band_part(self.w, 0, -1)
        tensor_args = tf.concat(args, axis=1)
        X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                       tensor_args], 1)
        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [4, 1, 1]), W)
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])
        gX = tf.matmul(tf.tanh(XWX), self.u)
        result = tf.sigmoid(gX)
        return result
