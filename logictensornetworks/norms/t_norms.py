import tensorflow as tf

'''########################## Min norms ###################################'''


@tf.function
def min_and(wff):
    return tf.math.reduce_min(wff, axis=-1, keepdims=True)


@tf.function
def min_or(wff):
    return tf.math.reduce_max(wff, axis=-1, keepdims=True)


@tf.function
def min_not(wff):
    return 1 - wff


@tf.function
def min_implies(wff1, wff2):
    leq = tf.cast(wff1 <= wff2, dtype=tf.dtypes.float32)  # should be useless
    min = tf.math.minimum(wff1, wff2)

    return tf.math.maximum(leq, min)


@tf.function
def min_equivalent(wff1, wff2):
    eq = tf.cast(wff1 == wff2, dtype=tf.dtypes.float32)
    min = tf.math.minimum(wff1, wff2)

    return tf.math.maximum(eq, min)


'''########################## Lukasiewicz norms ###################################'''


@tf.function
def luk_and(wff):
    a = tf.math.reduce_sum(wff, axis=-1, keepdims=True)
    b = tf.cast(tf.shape(wff)[-1], dtype=tf.dtypes.float32)

    return tf.math.maximum(0.0, 1 + a - b, axis=-1, keepdims=True)


@tf.function
def luk_or(wff):
    return tf.math.minimum(tf.math.reduce_sum(wff, axis=-1, keepdims=True), 1.0)


@tf.function
def luk_not(wff):
    return 1 - wff


@tf.function
def luk_implies(wff1, wff2):
    return tf.math.minimum(1., 1 - wff1 + wff2)


@tf.function
def luk_equivalent(wff1, wff2):
    return 1 - abs(wff1 - wff2)


'''########################## Mean norms ###################################'''


@tf.function
def mean_and(wffs):
    return tf.math.reduce_mean(wffs, axis=-1, keepdims=True)


@tf.function
def mean_or(wffs):
    return tf.math.reduce_max(wffs, axis=-1, keepdims=True)


@tf.function
def mean_implies(wff1, wff2):
    return tf.clip_by_value(2 * wff2 - wff1, 0, 1)


@tf.function
def mean_not(wff):
    return 1 - wff


@tf.function
def mean_equivalent(wff1, wff2):
    return 1 - tf.abs(wff1 - wff2)


'''########################## Prod norms ###################################'''


@tf.function
def prod_and(wffs):
    return tf.reduce_prod(wffs, axis=-1, keepdims=True)


@tf.function
def prod_or(wffs):
    return 1 - tf.reduce_prod(1 - wffs, axis=-1, keepdims=True)


@tf.function
def prod_implies(wff1, wff2):
    le_wff1_wff2 = tf.cast(wff1 <= wff2, dtype=tf.dtypes.float32)
    gt_wff1_wff2 = tf.cast(wff1 > wff2, dtype=tf.dtypes.float32)

    if wff1[0] == 0:
        return le_wff1_wff2 + gt_wff1_wff2 * wff2 / wff1
    else:
        return tf.constant([1.0])


@tf.function
def prod_not(wff):
    # according to standard goedel logic is
    # return tf.to_float(tf.equal(wff,1))
    return 1 - wff


@tf.function
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
