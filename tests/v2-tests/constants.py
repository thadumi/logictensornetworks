import tensorflow as tf


class constant:
    def __init__(self):
        self._vars = []

    @tf.function
    def __call__(self, value=None, min_value=None, max_value=None):
        if value is not None:
            result = tf.constant(value)
        else:
            self._vars.append(tf.Variable(
                tf.random.uniform(
                    shape=(1, len(min_value)),
                    minval=min_value,
                    maxval=max_value)))
            result = self._vars[-1]

        result.doms = []
        return result


size = 20

'''
const = constant()
c1 = const(min_value=[0.] * size, max_value=[1.] * size)
c2 = const(min_value=[0.] * size, max_value=[1.] * size)

print(c1)
print(c2)
'''


def constant_fun():
    constVar = None

    @tf.function
    def _const(value=None, min_value=None, max_value=None):

        if value is not None:
            result = tf.constant(value)
            result.doms = []
            return result
        else:
            nonlocal constVar
            if constVar is None:
                constVar = tf.Variable(
                    tf.random.uniform(
                        shape=(1, len(min_value)),
                        minval=min_value,
                        maxval=max_value))
            constVar.doms = []
            return constVar

    return _const


class constant1:
    def __init__(self):
        self._result = None

    def __call__(self, value=None, min_value=None, max_value=None):
        if value is not None:
            self._result = tf.constant(value)
        else:
            if self._result is None:
                self._result = tf.Variable(
                    tf.random.uniform(
                        shape=(1, len(min_value)),
                        minval=min_value,
                        maxval=max_value))
        self._result.doms = []
        return self._result


def const(value=None, min_value=None, max_value=None):
    if value is not None:
        result = tf.constant(value)
    else:
        result = tf.Variable(
            tf.random.uniform(
                shape=(1, len(min_value)),
                minval=min_value,
                maxval=max_value))
    result.doms = []
    return result


BIAS_factor = 0.0
BIAS = 0.0


def cross_args(args):
    result = args[0]
    for arg in args[1:]:
        result, _ = cross_2args(result, arg)
    result_flat = tf.reshape(result,
                             (tf.reduce_prod(tf.shape(result)[:-1]),
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
        mult_eX[i] = tf.maximum(1, tf.floor_div(tf.shape(eY)[i], tf.shape(eX)[i]))
        mult_eY[i] = tf.maximum(1, tf.floor_div(tf.shape(eX)[i], tf.shape(eY)[i]))
    result1 = tf.tile(eX, mult_eX)
    result2 = tf.tile(eY, mult_eY)
    result = tf.concat([result1, result2], axis=-1)
    result1.doms = eX_doms
    result2.doms = eX_doms
    result.doms = eX_doms
    return result, [result1, result2]


g1 = {l: const(min_value=[0.] * size, max_value=[1.] * size) for l in 'abcdefgh'}


def variable(number_of_features_or_feed, label='variable'):
    if isinstance(number_of_features_or_feed, tf.Tensor):
        result = tf.identity(number_of_features_or_feed, name=label)
    else:
        result = tf.constant(number_of_features_or_feed, name=label)

    result.doms = [label]
    return result


def predicate(number_of_features_or_vars, pred_definition=None, layers=None, label='predicate'):
    layers = layers or 4
    global BIAS

    if type(number_of_features_or_vars) is list:  # list of vars I suppose
        number_of_features = sum([int(v.shape[1]) for v in number_of_features_or_vars])
    elif type(number_of_features_or_vars) is tf.Tensor:
        number_of_features = int(number_of_features_or_vars.shape[1])
    else:
        number_of_features = number_of_features_or_vars
    if pred_definition is None:
        # if there is not a custom predicate model defined create create the default schema
        W = tf.linalg.band_part(
            tf.Variable(
                tf.random.normal(
                    [layers,
                     number_of_features + 1,
                     number_of_features + 1], mean=0, stddev=1), name="W" + label), 0, -1)
        u = tf.Variable(tf.ones([layers, 1]),
                        name="u" + label)

        def apply_pred(*args):
            app_label = (label + "/" + "_".join([arg.name.split(":")[0] for arg in args]) + "/") \
                if not tf.executing_eagerly() else (label + '_applied_pred')

            tensor_args = tf.concat(args, axis=1)
            X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                           tensor_args], 1)
            XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [layers, 1, 1]), W)
            XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])
            gX = tf.matmul(tf.tanh(XWX), u)
            result = tf.sigmoid(gX, name=app_label)
            return result

        pars = [W, u]
    else:
        def apply_pred(*args):
            return pred_definition(*args)

        pars = []

    def pred(*args):
        global BIAS
        crossed_args, list_of_args_in_crossed_args = cross_args(args)
        result = apply_pred(*list_of_args_in_crossed_args)
        if crossed_args.doms != []:
            result = tf.reshape(result, tf.concat([tf.shape(crossed_args)[:-1], [1]], axis=0))
        else:
            result = tf.reshape(result, (1,))
        result.doms = crossed_args.doms
        BIAS = tf.divide(BIAS + .5 - tf.reduce_mean(result), 2) * BIAS_factor
        return result

    pred.pars = pars
    pred.label = label
    return pred


print('\n\n\n')
var = variable(tf.concat(list(g1.values()), axis=0))

Friends = predicate(size * 2, label='Friends')
print(Friends.pars)

print(Friends(g1['a'], g1['b']))
