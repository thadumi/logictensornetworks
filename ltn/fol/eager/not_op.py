"""
:Date: Nov 21, 2019
:Version: 0.0.3
"""

from eager.base_operation import LtnOperation
from ltn.backend.norms import F_Not


class LtnNot(LtnOperation):

    def __init__(self,
                 tensors_meta=None):
        if tensors_meta is None:
            tensors_meta = ''

        super(LtnNot, self).__init__(op_name='NOT_' + tensors_meta)

    def call(self, *doms, **kwargs):
        # @tf.function
        def not_op(*args):
            return F_Not(args[0])

        return not_op, doms[0]


def Not(*tensors):
    tensors_meta = tensors[0]._ltn_op._ltn_op_name
    return LtnNot(tensors_meta=tensors_meta)(*tensors)