"""
:Date: Nov 19, 2019
:Version: 0.0.1
"""

import tensorflow as tf

from ltn.backend.ltn_utils import cross_args, cross_args_doms
from ltn.fol.base_operation import LtnOperation
from ltn.norms.norms_config import F_Or


class LtnOr(LtnOperation):

    def __init__(self,
                 tensors_meta=None):
        if tensors_meta is None:
            tensors_meta = ['']
        label = '_OR_'.join([meta for meta in tensors_meta])

        super(LtnOr, self).__init__(op_name=label)

    @tf.function
    def call(self, *args, **kwargs):
        if len(args) == 0:
            return tf.constant(0.0)
        else:
            cross_wffs, _ = cross_args(args)
            return tf.identity(F_Or(cross_wffs), name=self._ltn_op_name)

    def compute_doms(self, *args, **kwargs):
        return [] if len(args) == 0 else cross_args_doms(doms=[arg._ltn_doms for arg in args])


def Or(*tensors):
    tensors_meta = [tensor._ltn_op._ltn_op_name for tensor in tensors]
    return LtnOr(tensors_meta=tensors_meta)(*tensors)
