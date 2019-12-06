"""
:Date: Dec 03, 2019
:Version: 0.0.3
"""
import functools
import logging

import tensorflow as tf


# assuming args are just the doms
# TODO: WARN(thadumi) doms could not be flat
def cross_args(*doms):
    # a list is not hashable for lru but a tuple is
    return _cross_args(tuple(doms))


@functools.lru_cache(maxsize=None)
def _cross_args(doms):
    logging.debug('cross_args: computing a new tensor_cross_args for' + str(doms))
    result_doms = doms[0]

    lambdas = []
    for y_dom in doms[1:]:
        result_doms, _cross_2args = cross_2args(result_doms, y_dom)
        lambdas.append(_cross_2args)

    def tensor_cross_args(*tensors):
        result_tensor = tensors[0]
        for i in range(len(lambdas)):
            result_tensor = lambdas[i](result_tensor, tensors[i + 1])

        return result_tensor

    return result_doms, tensor_cross_args


def split_cross_args(result_tensor, *tensors):
    result_flat = tf.reshape(result_tensor,
                             (tf.math.reduce_prod(tf.shape(result_tensor)[:-1]),
                              tf.shape(result_tensor)[-1]))

    result_args = tf.split(result_flat, [tf.shape(tensor)[-1] for tensor in tensors], 1)
    return result_args


def split_cross_2args(result_tensor):
    return tf.split(result_tensor, 2, axis=-1)


def tensors_cross_2args_default(X, Y):
    return tf.concat([X, Y], axis=-1)


def expands_x(X, number_of_times_x_expands=0):
    '''
        def condition(i, args):
            return tf.greater(i, 0)

        def body(i, value):
            new_value = value[tf.newaxis, :]
            return tf.subtract(i, 1), new_value

    return tf.while_loop(condition, body,
                             (number_of_times_x_expands, X),
                             shape_invariants=(number_of_times_x_expands.get_shape(),
                                               tf.TensorShape([None] + X.shape.as_list())))

    '''

    tmp_X = X
    for _ in range(number_of_times_x_expands):
        tmp_X = tmp_X[tf.newaxis, :]

    return tmp_X


def expands_y(Y, number_of_times_y_expands=10):
    tmp_Y = Y
    for _ in range(number_of_times_y_expands):
        tmp_Y = tf.expand_dims(tmp_Y, -2)
    return tmp_Y


@functools.lru_cache(maxsize=None)
def cross_2args(x_ltn_doms, y_ltn_doms):
    logging.debug('cross_2args: computing a new tensor_cross_args for' + str((x_ltn_doms, y_ltn_doms)))

    if (x_ltn_doms is None or x_ltn_doms == ()) and (y_ltn_doms is None or y_ltn_doms == ()):
        return tuple(), tensors_cross_2args_default

    X_Y = set(x_ltn_doms) - set(y_ltn_doms)
    Y_X = set(y_ltn_doms) - set(x_ltn_doms)

    # eX = X
    eX_doms = [x for x in x_ltn_doms]
    for y in Y_X:
        eX_doms = [y] + eX_doms
    number_of_times_x_expands = len(Y_X)

    # eY = Y
    eY_doms = [y for y in y_ltn_doms]
    for x in X_Y:
        eY_doms.append(x)
    number_of_times_y_expands = len(X_Y)

    perm_eY = []
    for y in eY_doms:
        perm_eY.append(eX_doms.index(y))
    default_perm = perm_eY + [len(perm_eY)]

    mult_eX_size = len(eX_doms) + 1
    mult_eY_size = len(eY_doms) + 1

    @tf.function
    def tensors_cross_2args(X, Y):
        eX = expands_x(X, number_of_times_x_expands=number_of_times_x_expands)
        eY = expands_y(Y, number_of_times_y_expands=number_of_times_y_expands)
        eY = tf.transpose(eY, perm=default_perm)

        mult_eX = [1] * mult_eX_size
        mult_eY = [1] * mult_eY_size

        for i in range(len(mult_eX) - 1):
            mult_eX[i] = tf.math.maximum(1, tf.math.floordiv(tf.shape(eY)[i], tf.shape(eX)[i]))
            mult_eY[i] = tf.math.maximum(1, tf.math.floordiv(tf.shape(eX)[i], tf.shape(eY)[i]))

        return tf.concat([tf.tile(eX, mult_eX), tf.tile(eY, mult_eY)], axis=-1)

    return tuple(eX_doms), tensors_cross_2args
