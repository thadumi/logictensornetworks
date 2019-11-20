"""
:Date: Nov 19, 2019
:Version: 0.0.1
"""

import tensorflow as tf

from ltn.fol.base_operation import LtnOperation
from ltn.norms.norms_config import F_ForAll


class LtnForall(LtnOperation):

    def __init__(self):
        super(LtnForall, self).__init__(op_name='Forall')

    def call(self, args, wff, **kwargs):

        result_doms = [x for x in wff._ltn_doms if x not in [var._ltn_doms[0] for var in args]]
        quantif_axis = [wff._ltn_doms.index(var._ltn_doms[0]) for var in args]

        not_empty_vars = tf.cast(
            tf.math.reduce_prod(tf.stack([tf.size(var) for var in args])),
            dtype=tf.dtypes.bool)

        ones = tf.ones((1,) * (len(result_doms) + 1))

        # if not_empty_vars:
        #    result = F_ForAll(quantif_axis, wff)
        # else:
        #    result = ones

        return tf.cond(not_empty_vars, lambda: F_ForAll(quantif_axis, wff), lambda: ones)

    def compute_doms(self, args, wff, **kwargs):
        return [x for x in wff._ltn_doms if x not in [var._ltn_doms[0] for var in args]]


def Forall(args, wff):
    if type(args) is not tuple and type(args) is not list:
        args = [args, ]

    return LtnForall()(args, wff)
