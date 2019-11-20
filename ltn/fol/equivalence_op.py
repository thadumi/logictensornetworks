"""
:Date: Nov 19, 2019
:Version: 0.0.1
"""

import tensorflow as tf

from ltn.backend.ltn_utils import cross_2args_doms, cross_2args
from ltn.fol.base_operation import LtnOperation
from ltn.norms.norms_config import F_Equiv


class LtnEquivalence(LtnOperation):

    def __init__(self, tensors_meta=None):
        if tensors_meta is None:
            tensors_meta = ['']
        label = tensors_meta[0] + '_IFF_' + tensors_meta[1]

        super(LtnEquivalence, self).__init__(op_name=label)

    def call(self, arg1, arg2, **kwargs):
        _, cross_wffs = cross_2args(arg1, arg2)

        result = F_Equiv(cross_wffs[0], cross_wffs[1])
        return tf.identity(result, name=self._ltn_op_name)

    def compute_doms(self, *args, **kwargs):
        return cross_2args_doms(args[0]._ltn_doms, args[1]._ltn_doms)[0]


def Equiv(*tensors):
    tensors_meta = [tensor._ltn_op._ltn_op_name for tensor in tensors]
    return LtnEquivalence(tensors_meta=tensors_meta)(*tensors)
