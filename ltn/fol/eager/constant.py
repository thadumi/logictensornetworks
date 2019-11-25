"""
:Date: Nov 21, 2019
:Version: 0.0.3
"""

import tensorflow as tf

import ltn.backend.fol_status as FOL
from eager.base_operation import LtnOperation


class LtnConstant(LtnOperation):
    def __init__(self, const_name):
        super(LtnConstant, self).__init__(op_name='C_' + const_name, domain=[])
        self._const = const_name

    def call(self, value=None,
             min_value=None, max_value=None,
             **kwargs):
        label = self._ltn_op_name
        if value is not None:
            # @tf.function
            def const():
                return tf.constant(value, name=label)
        else:
            const_tensor = self._add_weight(tf.random.uniform(shape=(1, len(min_value)),
                                                              minval=min_value,
                                                              maxval=max_value),
                                            name=label)

            # @tf.function
            def const():
                return const_tensor

        return const, []

    def update_ltn_global_status(self, result_doms, output):
        FOL.CONSTANTS[self._const] = output

    @property
    def const(self):
        return self._const


def constant(label,
             value=None,
             min_value=None,
             max_value=None):
    return LtnConstant(label)(value=value, min_value=min_value, max_value=max_value)
