"""
:Date: Nov 19, 2019
:Version: 0.0.1
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
        # needs domain to be fixed?
        # cross_args
        output = self.call(*args, **kwargs)

        # TODO(thadumi) add a history track
        self._ltn_doms = self.compute_doms(*args, **kwargs)

        output._ltn_op = self  # store the instance of the operation which generated the output
        output._ltn_doms = self._ltn_doms

        return output

    def call(self, *args, **kwargs):
        raise Exception('exec operation not implemented')

    def _add_weight(self, *args, **kwargs):
        w = tf.Variable(*args, **kwargs)
        self._weights.append(w)

        # TODO(thadumi) move into FOL with a function like track_variable(var, owner)
        if self._ltn_op_name not in FOL._TF_VARIABLES.keys():
            FOL._TF_VARIABLES[self._ltn_op_name] = []
        FOL._TF_VARIABLES[self._ltn_op_name].append(w)

        return w

    def compute_doms(self, *args, **kwargs):
        return self._ltn_doms

    @property
    def doms(self):
        return self._ltn_doms

    @property
    def variables(self):
        return self._weights
