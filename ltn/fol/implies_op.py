"""
:Date: Nov 21, 2019
:Version: 0.0.3
"""

import tensorflow as tf

from ltn.backend.ltn_utils import cross_2args
from ltn.fol.base_operation import LtnOperation
from ltn.norms.norms_config import F_Implies


class LtnImplies(LtnOperation):

    def __init__(self,
                 tensors_meta=None):
        if tensors_meta is None:
            tensors_meta = ['']
        label = tensors_meta[0] + '_IMP_' + tensors_meta[1]

        super(LtnImplies, self).__init__(op_name=label)

    def call(self, arg1_doms, arg2_doms, **kwargs):
        eX_doms, _, _, tensor_cross_2args = cross_2args(x_ltn_doms=arg1_doms, y_ltn_doms=arg2_doms)

        # @tf.function
        def implies_op(arg1, arg2):
            _, cross_wffs = tensor_cross_2args(arg1, arg2)

            result = F_Implies(cross_wffs[0], cross_wffs[1])
            return tf.identity(result)

        return implies_op, eX_doms


def Implies(*tensors):
    tensors_meta = [tensor._ltn_op._ltn_op_name for tensor in tensors]
    return LtnImplies(tensors_meta=tensors_meta)(*tensors)
