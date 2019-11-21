"""
:Date: Nov 21, 2019
:Version: 0.0.3
"""

import tensorflow as tf

import ltn.backend.fol_status as FOL
from ltn.fol.base_operation import LtnOperation


class LtnVariable(LtnOperation):
    def __init__(self, var_name):
        super(LtnVariable, self).__init__(op_name='V_' + var_name, domain=[var_name])
        self._var = var_name

    def call(self, *args, tensor=None, constants=None, **kwargs):
        label = self._ltn_op_name
        if tensor is not None:
            def var(*_args):
                return tf.identity(tensor, name=label)

        elif constants is not None:
            consts = constants if type(constants) is list or type(constants) is tuple else [constants, ]

            def var(*_args):
                return tf.concat(consts, axis=0, name=label)
        else:
            raise Exception('Trying to create a Variable without defining its value')

        return var, [self.var]

    @property
    def var(self):
        return self._var

    def update_ltn_global_status(self, result_doms, output):
        FOL.VARIABLES[self._var] = output


def variable(label, tensor=None, constants=None):
    return LtnVariable(label)(tensor=tensor, constants=constants)
