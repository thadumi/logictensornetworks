"""
:Date: Nov 20, 2019
:Version: 0.0.2
"""

import tensorflow as tf


def validate_label(label):
    if label is None or not isinstance(label, str):
        raise ValueError('The label of a constant must be defined as a string')


# assuming args are just the doms
# TODO: WARN(thadumi) doms could not be flat
def cross_args(doms):
    result_doms = doms[0]

    lambdas = []
    for y_dom in doms[1:]:
        result_doms, _, _, _cross_2args = cross_2args(x_ltn_doms=result_doms, y_ltn_doms=y_dom)
        lambdas.append(_cross_2args)

    @tf.function
    def tensor_cross_args(tensors):
        result_tensor = tensors[0]
        for i in range(len(lambdas)):
            cross = lambdas[i]
            result_tensor, _ = cross(result_tensor, tensors[i])

        result_flat = tf.reshape(result_tensor,
                                 (tf.math.reduce_prod(tf.shape(result_tensor)[:-1]),
                                  tf.shape(result_tensor)[-1]))

        result_args = tf.split(result_flat, [tf.shape(tensor)[-1] for tensor in tensors], 1)
        return result_tensor, result_args

    return tensor_cross_args, result_doms


def cross_2args(x_ltn_doms=None, y_ltn_doms=None):
    if (x_ltn_doms is None or x_ltn_doms == []) and (y_ltn_doms is None or y_ltn_doms == []):
        @tf.function
        def _cross_2args_default(X, Y):
            result = tf.concat([X, Y], axis=-1)
            return result, [X, Y]

        return x_ltn_doms, x_ltn_doms, [], _cross_2args_default

    X_Y = set(x_ltn_doms) - set(y_ltn_doms)
    Y_X = set(y_ltn_doms) - set(x_ltn_doms)

    number_of_x_expands = 0
    # eX = X
    eX_doms = [x for x in x_ltn_doms]
    for y in Y_X:
        # eX = tf.expand_dims(eX, 0)
        eX_doms = [y] + eX_doms
    number_of_times_x_expands = len(Y_X)

    @tf.function
    def expands_x(X):
        tmp_X = X
        for _ in range(number_of_times_x_expands):
            tmp_X = tf.expand_dims(tmp_X, 0)
        return tmp_X

    # eY = Y
    eY_doms = [y for y in y_ltn_doms]
    for x in X_Y:
        # eY = tf.expand_dims(eY, -2)di
        eY_doms.append(x)
    number_of_times_y_expands = len(X_Y)

    @tf.function
    def expands_y(Y):
        tmp_Y = Y
        for _ in range(number_of_times_y_expands):
            tmp_Y = tf.expand_dims(tmp_Y, -2)
        return tmp_Y

    perm_eY = []
    for y in eY_doms:
        perm_eY.append(eX_doms.index(y))

    # eY = tf.transpose(eY, perm=perm_eY + [len(perm_eY)])
    def transpose_y(Y):
        # tf.print(tf.shape(Y), perm_eY, perm_eY + [len(perm_eY)])
        return tf.transpose(Y, perm=perm_eY + [len(perm_eY)])

    mult_eX = [1] * (len(eX_doms) + 1)
    mult_eY = [1] * (len(eY_doms) + 1)

    @tf.function
    def _cross_2args(X, Y):
        eX = expands_x(X)
        # tf.print(tf.shape(Y))
        eY = expands_y(Y)
        # tf.print(tf.shape(eY))
        eY = transpose_y(eY)

        for i in range(len(mult_eX) - 1):
            mult_eX[i] = tf.math.maximum(1, tf.math.floordiv(tf.shape(eY)[i], tf.shape(eX)[i]))
            mult_eY[i] = tf.math.maximum(1, tf.math.floordiv(tf.shape(eX)[i], tf.shape(eY)[i]))

        result1 = tf.tile(eX, mult_eX)
        result2 = tf.tile(eY, mult_eY)
        result = tf.concat([result1, result2], axis=-1)
        return result, [result1, result2]

    return eX_doms, eY_doms, perm_eY, _cross_2args