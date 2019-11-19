"""
:Date: Nov 18, 2019
:Version: 0.0.1
"""

import tensorflow as tf

import ltn.backend.fol_status as FOL
from ltn.fol.base_operation import LtnOperation


class LtnVariable(LtnOperation):
    def __init__(self, var_name):
        super(LtnVariable, self).__init__(op_name='Variable_' + var_name, domain=[var_name])
        self._var = var_name

    def call(self, tensor=None, constants=None):
        label = self._ltn_op_name

        if tensor is not None:
            var = tf.identity(tensor, name=label)
        elif constants is not None:
            if type(constants) is not list:
                constants = [constants, ]

            var = tf.concat(constants, axis=0, name=label)

        FOL.VARIABLES[self._var] = var
        return var

    @property
    def var(self):
        return self._var


def variable(label, tensor=None, constants=None):
    return LtnVariable(label)(tensor=tensor, constants=constants)
