"""
:Date: Nov 21, 2019
:Version: 0.0.3
"""

import tensorflow as tf

import ltn.backend.fol_status as FOL


class LtnOperation(object):

    def __init__(self,
                 op_name=None,
                 domain=None):
        self._ltn_op_name = op_name
        self._ltn_doms = domain

        self._weights = []

    def __call__(self, *args, **kwargs):
        doms = []
        for arg in args:
            if type(arg) is tuple or type(arg) is list:
                doms.append([_arg._ltn_doms for _arg in arg])
            else:
                doms.append(arg._ltn_doms)

        # TODO/MEMO(thadumi): this could be a list of lambdas not just one
        # kwargs['tensor_cross_args'] = tensor_cross_args
        # kwargs['result_doms'] = result_doms  # TODO(thadumi) should split into two calls like cross_args

        tensor_computation, result_doms = self.call(*doms, **kwargs)
        output = tensor_computation(*args)

        # TODO(thadumi) add a history track
        self.update_ltn_global_status(result_doms, output)

        # self._ltn_doms = result_doms
        output._ltn_op = self  # store the instance of the operation which generated the output
        output._ltn_doms = result_doms

        return output

    def call(self, *doms, **kwargs):
        """
        :param doms: the doms arguments
        :param kwargs: external kwargs given by the user plus the kwargs
        :return: the new doms and a lambda containing the computation which needs to be done on the real arguments (aka tensors)
        """
        raise Exception('call operation not implemented')

    def _add_weight(self, *args, **kwargs):
        w = tf.Variable(*args, **kwargs)
        self._weights.append(w)

        # TODO(thadumi) move into FOL with a function like track_variable(var, owner)
        #if self._ltn_op_name not in FOL._TF_VARIABLES.keys():
        #    FOL._TF_VARIABLES[self._ltn_op_name] = []
        FOL._TF_VARIABLES.append(w)

        return w

    def update_ltn_global_status(self, *args, **kwargs):
        pass

    @property
    def doms(self):
        return self._ltn_doms

    @property
    def variables(self):
        return self._weights
