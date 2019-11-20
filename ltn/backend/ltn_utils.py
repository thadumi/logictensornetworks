"""
:Date: Nov 18, 2019
:Version: 0.0.1
"""

import tensorflow as tf

def validate_label(label):
    if label is None or not isinstance(label, str):
        raise ValueError('The label of a constant must be defined as a string')


def cross_args(args):
    result = args[0]

    for arg in args[1:]:
        result, _ = cross_2args(result, arg)

    result_flat = tf.reshape(result,
                             (tf.math.reduce_prod(tf.shape(result)[:-1]),
                              tf.shape(result)[-1]))

    result_args = tf.split(result_flat, [tf.shape(arg)[-1] for arg in args], 1)
    return result, result_args


def cross_2args(X, Y):
    if X._ltn_doms == [] and Y._ltn_doms == []:
        result = tf.concat([X, Y], axis=-1)
        result._ltn_doms = []
        return result, [X, Y]

    X_Y = set(X._ltn_doms) - set(Y._ltn_doms)
    Y_X = set(Y._ltn_doms) - set(X._ltn_doms)

    eX = X
    eX_doms = [x for x in X._ltn_doms]
    for y in Y_X:
        eX = tf.expand_dims(eX, 0)
        eX_doms = [y] + eX_doms

    eY = Y
    eY_doms = [y for y in Y._ltn_doms]
    for x in X_Y:
        eY = tf.expand_dims(eY, -2)
        eY_doms.append(x)

    perm_eY = []
    for y in eY_doms:
        perm_eY.append(eX_doms.index(y))

    eY = tf.transpose(eY, perm=perm_eY + [len(perm_eY)])
    mult_eX = [1] * (len(eX_doms) + 1)
    mult_eY = [1] * (len(eY_doms) + 1)

    for i in range(len(mult_eX) - 1):
        mult_eX[i] = tf.math.maximum(1, tf.math.floordiv(tf.shape(eY)[i], tf.shape(eX)[i]))
        mult_eY[i] = tf.math.maximum(1, tf.math.floordiv(tf.shape(eX)[i], tf.shape(eY)[i]))

    result1 = tf.tile(eX, mult_eX)
    result2 = tf.tile(eY, mult_eY)
    result = tf.concat([result1, result2], axis=-1)

    result1._ltn_doms = eX_doms
    result2._ltn_doms = eX_doms
    result._ltn_doms = eX_doms

    return result, [result1, result2]


def cross_args_doms(doms):
    dom0 = doms[0]

    for dom in doms[1:]:
        dom0, _ = cross_2args_doms(dom0, dom)

    return dom0

def cross_2args_doms(x_doms, y_doms):
    if x_doms == [] and y_doms == []:
        return [], []

    X_Y = set(x_doms) - set(y_doms)
    Y_X = set(y_doms) - set(x_doms)

    eX = [x for x in x_doms]
    for y in Y_X:
        eX = [y] + eX

    eY = [y for y in y_doms]
    for x in X_Y:
        eY.append(x)

    '''perm_eY = []
    for y in eY:
        perm_eY.append(eX.index(y))
    '''
    return eX, eY