"""
:Date: Nov 19, 2019
:Version: 0.0.1
"""

import tensorflow as tf

from ltn.fol.base_operation import LtnOperation
from ltn.norms.norms_config import F_Not


class LtnNot(LtnOperation):

    def __init__(self,
                 tensors_meta=None):
        if tensors_meta is None:
            tensors_meta = ''

        super(LtnNot, self).__init__(op_name='NOT_' + tensors_meta)

    def call(self, *args, **kwargs):
        return tf.identity(F_Not(args[0]), name=self._ltn_op_name)

    def compute_doms(self, *args, **kwargs):
        return args[0]._ltn_doms


def Not(*tensors):
    tensors_meta = tensors[0]._ltn_op._ltn_op_name
    return LtnNot(tensors_meta=tensors_meta)(*tensors)
