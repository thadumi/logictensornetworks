"""
:Author: Theodor A. Dumitrescu
:Date: 16/12/19
:Version: 0.0.1
"""

import functools
import logging
from typing import Optional, Callable, Tuple

import tensorflow as tf

import ltn.fol.fol_status as FOL
from logic import LogicalComputation
from ltn.backend.utils import cross_args, split_cross_args, _cross_args


class LogicalFunction(LogicalComputation):
    def __init__(self,
                 function,
                 input_doms=None,
                 output_doms=None,
                 input_terms=None,
                 function_computation=None):
        super(LogicalFunction, self).__init__(input_doms, output_doms, input_terms)
        self.function = function
        self.function_computation = function_computation

    def _compute(self, *args) -> tf.Tensor:
        return self.function_computation(*args)

    def __str__(self):
        return self.function.name + '(' + ', '.join([str(i) for i in self._ltn_args]) + ')'


class Function(object):
    def __init__(self, **kwargs):
        logging.info('Clearing a new predicate: ' + str(kwargs))

        self.name = kwargs['name']
        self.function_definition = kwargs['function_definition']
        self.output_shape_spec = kwargs['output_shape_spec']
        self.number_of_features = kwargs['number_of_features']

        super(Function, self).__init__()

    def __call__(self, *args, **kwargs):
        input_doms = [arg.doms for arg in args]
        output_doms, tensor_cross_args = cross_args(*input_doms)

        return LogicalFunction(self,
                               input_doms, output_doms,
                               [*args],
                               _function_computational(tensor_cross_args,
                                                       self.function_definition,
                                                       self.output_shape_spec,
                                                       bool(output_doms)))


def function(name: str,
             input_shape_spec: int,
             output_shape_spec: Tuple[int, ...] = (1,),
             function_definition: Optional[Callable[[tf.Tensor, ], tf.Tensor]] = None):

    if FOL.function_already_defined(name):
        msg = '[function] There is already a function having the name `{}`'.format(name)
        logging.error(msg)

        raise ValueError(msg)

    number_of_features = input_shape_spec

    if function_definition is None:
        function_definition = DefaultFunctionModel(number_of_features, output_shape_spec)


    config = {'name': name,
              'function_definition': function_definition,
              'output_shape_spec': output_shape_spec,
              'number_of_features': number_of_features}

    f = Function(**config)
    FOL.track_function(name, f)
    return f


class DefaultFunctionModel(object):
    def __init__(self, number_of_features, output_shape_spec):
        self.W = FOL.variable(
            tf.random.normal(
                [number_of_features + 1, output_shape_spec],
                mean=0,
                stddev=1)
        )

    @tf.function
    def __call__(self, *args, **kwargs):
        tensor_args = tf.concat(args, axis=1)
        X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                       tensor_args], 1)
        result = tf.matmul(X, self.W)

        return result


@functools.lru_cache(maxsize=None)
def _function_computational(tensor_cross_args: Callable,
                            function_definition: Callable,
                            output_shape: int,
                            need_reshape: bool) -> Callable:
    @tf.function
    def _computation(*args):
        crossed_args = tensor_cross_args(*args)
        result = function_definition(*split_cross_args(crossed_args, *args))

        if need_reshape:
            return tf.reshape(result, tf.concat([tf.shape(crossed_args)[:-1], tf.shape(result)[-1:]],
                                                axis=0))
        else:
            return tf.reshape(result, output_shape)

    return _computation
