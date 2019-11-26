"""
:Author: Theodor A. Dumitrescu
:Date: 26/11/19
:Version: 0.0.2
"""
import tensorflow as tf

'''########################## Min norms ###################################'''


def min_and(wff):
    return tf.math.reduce_min(wff, axis=-1, keepdims=True)


def min_or(wff):
    return tf.math.reduce_max(wff, axis=-1, keepdims=True)


def min_not(wff):
    return 1 - wff


def min_implies(wff1, wff2):
    return tf.math.maximum(tf.cast(tf.math.less_equal(wff1, wff2), dtype=tf.dtypes.float32),
                           wff2)


def min_equivalent(wff1, wff2):
    return tf.math.maximum(tf.cast(tf.math.equal(wff1, wff2), dtype=tf.float32),
                           tf.math.minimum(wff1, wff2))


'''########################## Lukasiewicz norms ###################################'''


def luk_and(wff):
    a = tf.math.reduce_sum(wff, axis=-1, keepdims=True)
    b = tf.cast(tf.shape(wff)[-1], dtype=tf.dtypes.float32)

    c = tf.cast(tf.shape(wff)[-1], dtype=tf.dtypes.float32)

    # return tf.math.maximum(0.0, tf.math.reduce_sum(wff, axis=-1, keepdims=True) + 1 - c)
    return tf.math.maximum(0.0,
                           tf.math.reduce_sum(input_tensor=wff, axis=-1, keepdims=True)
                           + 1
                           - tf.cast(tf.shape(input=wff)[-1], dtype=tf.float32))


def luk_or(wff):
    return tf.math.minimum(tf.math.reduce_sum(wff, axis=-1, keepdims=True), 1.0)


def luk_not(wff):
    return 1 - wff


def luk_implies(wff1, wff2):
    return tf.math.minimum(1., 1 - wff1 + wff2)


def luk_equivalent(wff1, wff2):
    return 1 - tf.math.abs(wff1 - wff2)


'''########################## Mean norms ###################################'''


def mean_and(wffs):
    return tf.math.reduce_mean(wffs, axis=-1, keepdims=True)


def mean_or(wffs):
    return tf.math.reduce_max(wffs, axis=-1, keepdims=True)


def mean_implies(wff1, wff2):
    return tf.clip_by_value(2 * wff2 - wff1, 0, 1)


def mean_not(wff):
    return 1 - wff


def mean_equivalent(wff1, wff2):
    return 1 - tf.math.abs(wff1 - wff2)


'''########################## Prod norms ###################################'''


def prod_and(wffs):
    return tf.reduce_prod(wffs, axis=-1, keepdims=True)


def prod_or(wffs):
    return 1 - tf.reduce_prod(1 - wffs, axis=-1, keepdims=True)


def prod_implies(wff1, wff2):
    le_wff1_wff2 = tf.cast(tf.math.less_equal(wff1, wff2), dtype=tf.dtypes.float32)
    gt_wff1_wff2 = tf.cast(tf.math.greater(wff1, wff2), dtype=tf.dtypes.float32)

    return tf.cond(pred=tf.math.equal(wff1[0], 0),
                   true_fn=lambda: le_wff1_wff2 + gt_wff1_wff2 * wff2 / wff1,
                   false_fn=lambda: tf.constant([1.0]))


def prod_not(wff):
    # according to standard goedel logic is
    # return tf.to_float(tf.equal(wff,1))
    return 1 - wff


def prod_equivalent(wff1, wff2):
    return tf.math.minimum(wff1 / wff2, wff2 / wff1)


'''#####################################################################'''
'''############################ Norms For FOL ##########################'''
'''#####################################################################'''

'''########################## hmean norms ###################################'''


def hmean_universal_aggregation(axis, wff):
    return 1 / tf.reduce_mean(1 / (wff + 1e-10), axis=axis)


def min_universal_aggregation(axis, wff):
    return tf.reduce_min(wff, axis=axis)


def mean_universal_aggregation(axis, wff):
    return tf.reduce_mean(wff, axis=axis)


def max_existence_aggregation(axis, wff):
    return tf.reduce_max(wff, axis=axis)


# NORMS dic
TRIANGULAR_NORMS = {
    'min': {
        'AND': min_and,
        'OR': min_or,
        'NOT': min_not,
        'IMPLIES': min_implies,
        'EQUIVALENT': min_equivalent
    },

    'luk': {
        'AND': luk_and,
        'OR': luk_or,
        'NOT': luk_not,
        'IMPLIES': luk_implies,
        'EQUIVALENT': luk_equivalent
    },

    'prod': {
        'AND': luk_and,
        'OR': luk_or,
        'NOT': luk_not,
        'IMPLIES': luk_implies,
        'EQUIVALENT': luk_equivalent,

    },

    'mean': {
        'AND': mean_and,
        'OR': mean_or,
        'NOT': mean_not,
        'IMPLIES': mean_implies,
        'EQUIVALENT': mean_equivalent,
    },

    # #####  FOR FOL #####

    'universal': {
        'hmean': hmean_universal_aggregation,
        'min': min_universal_aggregation,
        'mean': mean_universal_aggregation,
    },

    'existence': {
        'max': max_existence_aggregation,
    }
}
