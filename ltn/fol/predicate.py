"""
:Date: Nov 21, 2019
:Version: 0.0.3
"""

import tensorflow as tf

from ltn.backend.ltn_utils import cross_args
from ltn.fol.base_operation import LtnOperation


class LtnPredicate(LtnOperation):
    def __init__(self,
                 predicate_name,
                 number_of_features,
                 pred_definition=None,
                 layers=None):
        super(LtnPredicate, self).__init__(op_name='P_' + predicate_name)

        self._pred = predicate_name
        self._layers = layers or 4
        self._number_of_features = number_of_features

        if pred_definition is None:
            # use default predicate model
            pred_definition = self.__create_default_predicate_model()

        self._pred_definition = pred_definition

    def call(self, *doms, **kwargs):
        tensor_cross_args, result_doms = cross_args(doms)

        if result_doms:
            def reshape(result, crossed_args):
                return tf.reshape(result,
                                  tf.concat([tf.shape(crossed_args)[:-1], [1]],
                                            axis=0))
        else:
            def reshape(result, crossed_args):
                return tf.reshape(result, (1,))

        _pred_definition = self._pred_definition

        # @tf.function
        def predicate_op(*args):
            crossed_args, list_of_args_in_crossed_args = tensor_cross_args(args)
            result = _pred_definition(*list_of_args_in_crossed_args)
            return reshape(result, crossed_args)

        return predicate_op, result_doms

    @property
    def pred(self):
        return self._pred

    def __create_default_predicate_model(self):
        w = self._add_weight(
            tf.random.normal([self._layers, self._number_of_features + 1, self._number_of_features + 1],
                             mean=0,
                             stddev=1),
            name="W" + self._ltn_op_name)

        u = self._add_weight(tf.ones([self._layers, 1]), name="u" + self._ltn_op_name)

        # @tf.function
        def pred_definition(*args):
            W = tf.linalg.band_part(w, 0, -1)
            tensor_args = tf.concat(args, axis=1)
            X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                           tensor_args], 1)
            XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self._layers, 1, 1]), W)
            XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])
            gX = tf.matmul(tf.tanh(XWX), u)
            result = tf.sigmoid(gX)
            return result

        return pred_definition


def predicate(label, number_of_variables_or_features):
    if type(number_of_variables_or_features) is list:  # list of vars/consts
        number_of_features = sum([int(v.shape[1]) for v in number_of_variables_or_features])
    elif tf.is_tensor(number_of_variables_or_features):
        number_of_features = int(number_of_variables_or_features.shape[1])
    else:
        number_of_features = number_of_variables_or_features

    return LtnPredicate(label, number_of_features)
