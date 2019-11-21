"""
:Date: Nov 21, 2019
:Version: 0.0.3
"""

import tensorflow as tf

from ltn.fol.base_operation import LtnOperation
from ltn.norms.norms_config import F_ForAll


class LtnForall(LtnOperation):

    def __init__(self):
        super(LtnForall, self).__init__(op_name='Forall')

    def call(self, args_doms, wff_doms, tensor_cross_args=None, result_doms=None, **kwargs):
        if type(args_doms[0]) is not list:
            args = (args_doms,)

        result_doms = [x for x in wff_doms if x not in [arg_doms[0] for arg_doms in args_doms]]
        quantif_axis = [wff_doms.index(arg_doms[0]) for arg_doms in args_doms]

        ones_length = len(result_doms) + 1

        # @tf.function
        def forall_op(args, wff):
            not_empty_vars = tf.cast(tf.math.reduce_prod(tf.stack([tf.size(var) for var in args])),
                                     dtype=tf.dtypes.bool)
            if not_empty_vars:
                result = F_ForAll(quantif_axis, wff)
            else:
                result = tf.ones((1,) * ones_length)
            # tf.cond(not_empty_vars, lambda: F_ForAll(quantif_axis, wff), lambda: ones)

            return result

        return forall_op, result_doms


def Forall(args, wff):
    if type(args) is not tuple and type(args) is not list:
        args = [args, ]

    return LtnForall()(args, wff)
