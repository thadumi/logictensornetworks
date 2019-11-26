"""
:Author: thadumi
:Date: 26/11/19
:Version: 0.0.3
"""

import logging
from typing import Optional, List

import tensorflow as tf

import ltn.backend.fol_status as FOL
from ltn.fol.logic import LogicalComputation


class LogicalConstant(LogicalComputation):
    def __init__(self, **kwargs):
        super(LogicalConstant, self).__init__(None, [], [])
        self.name = kwargs['name']
        self.max_value = kwargs['max_value']
        self.min_value = kwargs['min_value']
        self.value = kwargs['value']

        self._tf_tensor = None

    def _compute(self, args):
        if self._tf_tensor is None:  # for lazy initialization of a constant
            if self.value is not None:
                self._tf_tensor = tf.constant(self.value)
            else:
                self._tf_tensor = FOL.variable(tf.random.uniform(shape=(1, len(self.min_value)),
                                                                 minval=self.min_value,
                                                                 maxval=self.max_value))
        return self._tf_tensor

    def __str__(self):
        return self.name


'''
    def _update_definition(self, definition):
        if tf.is_tensor(definition):
            tensor = definition
            definition = lambda *args: tensor

        self._definition = definition
'''


def constant(name: str,
             value: Optional[tf.Tensor] = None,
             min_value: Optional[List[float]] = None, max_value: Optional[List[float]] = None,
             size: Optional[int] = None) -> LogicalConstant:
    """
    TODO: add description of the constant
    :param name: the name of the constant. If the set of language constants already constantin one having the given name
                 an ValueError exception will be raised.
    :param value: the value of the constant. If the value is provided this constant will not change it's value during
                  the training
    :param min_value: TODO(thadumi) add doc the min_value argument
    :param max_value: TODO(thadumi) add doc the max_value argument
    :param size: number of features of a constant. Providing this parameter is equivalent of
                calling constant(name, min_value = [0.] * size, max_value = [1.] * size
    :return:
    """

    if FOL.constant_already_defined(name):
        msg = '[constant] There is already a constant having the name `{}`'.format(name)
        logging.error(msg)

        raise ValueError(msg)

    config = {'name': name, 'value': None, 'min_value': None, 'max_value': None}

    if value is not None:
        config['value'] = value
    elif size is not None:
        config['min_value'] = [0.] * size
        config['max_value'] = [1.] * size
    elif min_value is None and max_value is None:
        raise Exception('[constant] You should provide the complete value of the constant. '
                        'Otherwise, at least one of the two arguments `min_value` or `max_value` has to be provided.')
    else:
        if min_value is None:
            min_value = [0.] * len(max_value)
        if max_value is None:
            max_value = [1.] * len(min_value)

        config['min_value'] = min_value
        config['max_value'] = max_value

    logical_constant = LogicalConstant(**config)

    FOL.track_constant(name, logical_constant)
    return logical_constant


def _default_constant_definition(min_value=None, max_value=None):
    return tf.random.uniform(shape=(1, len(min_value)),
                             minval=min_value,
                             maxval=max_value)
