"""
:Author: thadumi
:Date: Dec 03, 2019
:Version: 0.0.5
"""
import functools
import logging
from typing import Optional, Callable

import tensorflow as tf

import ltn.backend.fol_status as FOL
from logic import LogicalComputation
from ltn.backend.ltn_utils import cross_args, split_cross_args, _cross_args


class LogicalPredicate(LogicalComputation):
    def __init__(self,
                 predicate,
                 input_doms=None,
                 output_doms=None,
                 input_terms=None,
                 predicate_computation=None):
        super(LogicalPredicate, self).__init__(input_doms, output_doms, input_terms)
        self.predicate = predicate
        self.predicate_computation = predicate_computation

    def _compute(self, *args) -> tf.Tensor:
        return self.predicate_computation(*args)

    def __str__(self):
        return self.predicate.name + '(' + ', '.join([str(i) for i in self._ltn_args]) + ')'


class Predicate(object):
    def __init__(self, **kwargs):
        logging.info('Clearing a new predicate: ' + str(kwargs))
        self.name = kwargs['name']
        self.predicate_definition = kwargs['predicate_definition']
        self.number_of_arguments = kwargs['number_of_arguments']
        self.argument_size = kwargs['argument_size']

    def __call__(self, *args: LogicalComputation, **kwargs) -> LogicalPredicate:
        input_doms = [arg.doms for arg in args]
        output_doms, tensor_cross_args = cross_args(*input_doms)
        logging.debug(_cross_args.cache_info())
        return LogicalPredicate(self,
                                input_doms, output_doms,
                                [*args],
                                _predicate_computational(tensor_cross_args,
                                                         self.predicate_definition,
                                                         bool(output_doms)))


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
    def __call__(self, *args: tf.Tensor) -> tf.Tensor:
        W = tf.linalg.band_part(self.w, 0, -1)

        tensor_args = tf.concat(args, axis=1)
        X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)), tensor_args], 1)

        XW = tf.linalg.matmul(tf.tile(tf.expand_dims(X, 0), [4, 1, 1]), W)
        XWX = tf.squeeze(tf.linalg.matmul(tf.expand_dims(X, 1), tf.transpose(XW, perm=[1, 2, 0])), axis=[1])
        gX = tf.linalg.matmul(tf.math.tanh(XWX), self.u)
        result = tf.math.sigmoid(gX)
        return result



@functools.lru_cache(maxsize=None)
def _predicate_computational(tensor_cross_args: Callable,
                             predicate_definition: Callable,
                             need_reshape: bool) -> Callable:
    @tf.function
    def _computation(*args):
        crossed_args = tensor_cross_args(*args)
        result = predicate_definition(*split_cross_args(crossed_args, *args))

        if need_reshape:
            return tf.reshape(result, tf.concat([tf.shape(input=crossed_args)[:-1], [1]], axis=0))
        else:
            return tf.reshape(result, (1,))

    return _computation
