"""
:Date: Nov 21, 2019
:Version: 0.0.3
"""

import tensorflow as tf

from ltn.backend.ltn_utils import cross_args
from ltn.fol.base_operation import LtnOperation
from ltn.norms.norms_config import F_Or


class LtnOr(LtnOperation):

    def __init__(self,
                 tensors_meta=None):
        if tensors_meta is None:
            tensors_meta = ['']
        label = '_OR_'.join([meta for meta in tensors_meta])

        super(LtnOr, self).__init__(op_name=label)

    def call(self, *doms, **kwargs):
        if len(doms) == 0:
            return lambda *args: tf.constant(0.0), []
        else:
            tensor_cross_args, result_doms = cross_args(doms)

            def or_op(*args):
                cross_wffs, _ = tensor_cross_args(args)
                return F_Or(cross_wffs)

            return or_op, result_doms


def Or(*tensors):
    tensors_meta = [tensor._ltn_op._ltn_op_name for tensor in tensors]
    return LtnOr(tensors_meta=tensors_meta)(*tensors)
