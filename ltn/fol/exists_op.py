"""
:Date: Nov 21, 2019
:Version: 0.0.3
"""

import tensorflow as tf

from ltn.fol.base_operation import LtnOperation
from ltn.norms.norms_config import F_Exists


class LtnExists(LtnOperation):

    def __init__(self):
        super(LtnExists, self).__init__(op_name='Exists')

    def call(self, vars_doms, wff_doms, **kwargs):
        if type(vars_doms[0]) is not list:
            vars_doms = (vars_doms,)

        result_doms = [x for x in wff_doms if x not in [var[0] for var in vars_doms]]
        quantif_axis = [wff_doms.index(var_doms[0]) for var_doms in vars_doms]

        zeros_length = len(result_doms) + 1

        # @tf.function
        def exists_op(vars, wff):
            not_empty_vars = tf.cast(tf.math.reduce_prod(tf.stack([tf.size(var) for var in vars])),
                                     dtype=tf.dtypes.bool)
            zeros = tf.zeros((1,) * zeros_length)

            if not_empty_vars:
                result = F_Exists(quantif_axis, wff)
            else:
                result = zeros

            return result

        return exists_op, result_doms


def Exists(args, wff):
    if type(args) is not tuple and type(args) is not list:
        args = [args, ]

    return LtnExists()(args, wff)
