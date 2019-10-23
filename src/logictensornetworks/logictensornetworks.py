import typing
import tensorflow as tf

from norms import TRIANGULAR_NORMS

BIAS_factor: float = 0.0
BIAS: float = 0.0
LAYERS: int = 4

F_And: typing.Callable[[typing.Any], typing.Any] = None
F_Or: typing.Callable[[typing.Any], typing.Any] = None
F_Not: typing.Callable[[typing.Any], typing.Any] = None

F_Implies: typing.Callable[[typing.Any, typing.Any], typing.Any] = None
F_Equiv: typing.Callable[[typing.Any, typing.Any], typing.Any] = None

F_ForAll: typing.Callable[[typing.Any, int], typing.Any] = None
F_Exists: typing.Callable[[typing.Any, int], typing.Any] = None


def set_tnorm(tnorm_kind: str):
    global F_And, F_Or, F_Implies, F_Not, F_Equiv
    assert tnorm_kind in TRIANGULAR_NORMS.keys()

    F_Or = TRIANGULAR_NORMS[tnorm_kind]['OR']
    F_And = TRIANGULAR_NORMS[tnorm_kind]['AND']
    F_Not = TRIANGULAR_NORMS[tnorm_kind]['NOT']
    F_Equiv = TRIANGULAR_NORMS[tnorm_kind]['EQUIVALENT']
    F_Implies = TRIANGULAR_NORMS[tnorm_kind]['IMPLIES']


def set_universal_aggregator(aggregator_kind: str):
    global F_ForAll
    assert aggregator_kind in TRIANGULAR_NORMS['universal'].keys()

    F_ForAll = TRIANGULAR_NORMS['universal'][aggregator_kind]


def set_existential_aggregator(aggregator_kind: str):
    global F_Exists
    assert aggregator_kind in TRIANGULAR_NORMS['universal'].keys()

    F_Exists = TRIANGULAR_NORMS['existence'][aggregator_kind]


def And(*wffs):
    if len(wffs) == 0:
        result = tf.constant(1.0)
        result.doms = []
    else:
        cross_wffs, _ = cross_args(wffs)
        label = "_AND_".join([wff.name.split(':'[0]) for wff in wffs])
        result = tf.identity(F_And(cross_wffs), name=label)

    return result


def Or(*wffs):
    if len(wffs) == 0:
        result = tf.constant(0.0)
        result.doms = []
    else:
        cross_wffs, _ = cross_args(wffs)
        label = "_OR_".join([wff.name.split(':')[0] for wff in wffs])
        result = tf.identity(F_Or(cross_wffs), name=label)
        result.doms = cross_wffs.doms

    return result


def Implies(wff1, wff2):
    _, cross_wffs = cross_2args(wff1, wff2)

    label = wff1.name.split(":")[0] + "_IMP_" + wff2.name.split(":")[0]
    result = F_Implies(cross_wffs[0], cross_wffs[1])
    result = tf.identity(result, name=label)
    result.doms = cross_wffs[0].doms
    return result


def Not(wff):
    result = F_Not(wff)
    label = "NOT_" + wff.name.split(":")[0]
    result = tf.identity(result, name=label)
    result.doms = wff.doms
    return result


def Equiv(wff1, wff2):
    _, cross_wffs = cross_2args(wff1, wff2)
    label = wff1.name.split(":")[0] + "_IFF_" + wff2.name.split(":")[0]

    result = F_Equiv(cross_wffs[0], cross_wffs[1])
    result = tf.identity(result, name=label)
    result.doms = cross_wffs[0].doms
    return result


@tf.function
def Forall(wff, args):
    if type(args) is not tuple:
        args = (args,)

    result_doms = [x for x in wff.doms if x not in [var.doms[0] for var in args]]
    quantif_axis = [wff.doms.index(var.doms[0]) for var in args]

    not_empty_vars = tf.cast(tf.math.reduce_prod(tf.stack([tf.size(var) for var in args])),
                             dtype=tf.dtypes.bool)

    ones = tf.ones((1,) * (len(result_doms) + 1))

    if not_empty_vars:
        result = F_ForAll(wff, quantif_axis)
    else:
        result = ones
    result.doms = result_doms

    return result


@tf.function
def Exists(vars, wff):
    if type(vars) is not tuple:
        vars = (vars,)

    result_doms = [x for x in wff.doms if x not in [var.doms[0] for var in vars]]
    quantif_axis = [wff.doms.index(var.doms[0]) for var in vars]

    not_empty_vars = tf.cast(tf.math.reduce_prod(tf.stack([tf.size(var) for var in vars])), dtype=tf.dtypes.bool)
    zeros = tf.zeros((1,) * (len(result_doms) + 1))

    if not_empty_vars:
        result = F_Exists(quantif_axis, wff)
    else:
        result = zeros
    result.doms = result_doms
    return result

@tf.function
def variable(placeholder, label, number_of_features_or_feed):
    if type(number_of_features_or_feed) is int:
        result = tf.placeholder(dtype=tf.float32, shape=(None, number_of_features_or_feed), name=label)
    elif isinstance(number_of_features_or_feed, tf.Tensor):
        result = tf.identity(number_of_features_or_feed, name=label)
    else:
        result = tf.constant(number_of_features_or_feed, name=label)
    result.doms = [label]
    return result


def constant(label, value=None, min_value=None, max_value=None):
    label = "ltn_constant_" + label
    if value is not None:
        result = tf.constant(value, name=label)
    else:
        result = tf.Variable(tf.random_uniform(
            shape=(1, len(min_value)),
            minval=min_value,
            maxval=max_value, name=label))
    result.doms = []
    return result


def function(label, input_shape_spec, output_shape_spec=1, fun_definition=None):
    pass


def proposition(label, initial_value=None, value=None):
    pass


def predicate(label, number_of_features_or_vars, pred_definition=None, layers=None):
    pass


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
    if X.doms == [] and Y.doms == []:
        result = tf.concat([X, Y], axis=-1)
        result.doms = []
        return result, [X, Y]

    X_Y = set(X.doms) - set(Y.doms)
    Y_X = set(Y.doms) - set(X.doms)

    eX = X
    eX_doms = [x for x in X.doms]
    for y in Y_X:
        eX = tf.expand_dims(eX, 0)
        eX_doms = [y] + eX_doms

    eY = Y
    eY_doms = [y for y in Y.doms]
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

    result1.doms = eX_doms
    result2.doms = eX_doms
    result.doms = eX_doms

    return result, [result1, result2]


'''DEFAULTS'''
set_tnorm("luk")
set_universal_aggregator("hmean")
set_existential_aggregator("max")
