"""
:Date: Nov 19, 2019
:Version: 0.0.1
"""

import tensorflow as tf

import ltn.backend.fol_status as FOL
from ltn.fol.base_operation import LtnOperation


class LtnConstant(LtnOperation):
    def __init__(self, const_name):
        super(LtnConstant, self).__init__(op_name='Constant_' + const_name, domain=[])
        self._const = const_name

    def call(self, value=None,
             min_value=None, max_value=None):
        label = self._ltn_op_name
        if value is not None:
            const = tf.constant(value, name=label)
        else:
            const = self._add_weight(tf.random.uniform(shape=(1, len(min_value)),
                                                       minval=min_value,
                                                       maxval=max_value),
                                     name=label)
        FOL.CONSTANTS[self._const] = const
        return const

    @property
    def const(self):
        return self._const


def constant(label,
             value=None,
             min_value=None,
             max_value=None):
    return LtnConstant(label)(value=value, min_value=min_value, max_value=max_value)
