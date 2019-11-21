"""
:Date: Nov 19, 2019
:Version: 0.0.1
"""

import tensorflow as tf

from ltn.fol.base_operation import LtnOperation
from ltn.norms.norms_config import F_Exists


class LtnExists(LtnOperation):

    def __init__(self):
        super(LtnExists, self).__init__(op_name='Exists')

    def call(self, *args, **kwargs):
        vars = args[0]
        wff = args[-1]


        #if type(vars) is not tuple or type(vars) is not list:
        #    vars = (vars,)

        result_doms = [x for x in wff._ltn_doms if x not in [var._ltn_doms[0] for var in vars]]
        quantif_axis = [wff._ltn_doms.index(var_doms) for var_doms in [var._ltn_doms[0] for var in vars]]

        not_empty_vars = tf.cast(tf.math.reduce_prod(tf.stack([tf.size(var) for var in vars])), dtype=tf.dtypes.bool)
        zeros = tf.zeros((1,) * (len(result_doms) + 1))

        if not_empty_vars:
            result = F_Exists(quantif_axis, wff)
        else:
            result = zeros
        return result

    def compute_doms(self, *args, **kwargs):
        wff = args[-1]
        vars = args[0]
        return [x for x in wff._ltn_doms if x not in [var._ltn_doms[0] for var in vars]]


def Exists(args, wff):
    if type(args) is not tuple and type(args) is not list:
        args = [args, ]

    return LtnExists()(args, wff)
