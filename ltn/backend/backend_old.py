"""
:Date: Nov 18, 2019
:Version: 0.0.1
"""
import tensorflow as tf

from tensorflow.python.eager import context

from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor

from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import array_ops

# The internal graph maintained by Keras and used by the symbolic Keras APIs
# while executing eagerly (such as the functional API for model-building).
_GRAPH = None

# A graph which is used for constructing functions in eager mode.
_CURRENT_SCRATCH_GRAPH = None

# This dictionary holds a mapping between a graph and variables to initialize
# in the graph.
_LTN_VARIABLES = set()


def get_graph():
    if context.executing_eagerly():
        global _GRAPH
        if _GRAPH is None:
            _GRAPH = func_graph.FuncGraph('ltn_graph')
        return _GRAPH
    else:
        return ops.get_default_graph()


def is_ltn_tensor(x):
    if not isinstance(x, (ops.Tensor,
                          variables_module.Variable,
                          sparse_tensor.SparseTensor)):
        raise ValueError('Unexpectedly found an instance of type `' + str(type(x)) +
                         '`. Expected a symbolic tensor instance.')
    return hasattr(x, '_doms')


def track_variable(v):
    """Tracks the given variable for initialization."""
    if context.executing_eagerly():
        return
    graph = v.graph if hasattr(v, 'graph') else get_graph()
    _LTN_VARIABLES.add(v)


def constant(value, name=None):
    if context.executing_eagerly():
        ltn_constant = tf.constant(value, name=name)
    else:
        with get_graph().as_default():
            ltn_constant = tf.constant(value, name=name)

    ltn_constant._doms = []
    return ltn_constant


def variable(value, dtype=None, name=None, constraint=None):
    ltn_v = tf.Variable(
        value,
        dtype=dtype,
        name=name,
        constraint=constraint)

    ltn_v._doms = [name]
    track_variable(ltn_v)

    return ltn_v


def placeholder(shape=None,
                ndim=None,
                dtype=None,
                name=None):
    if dtype is None:
        dtype = 'float32'
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    with get_graph().as_default():
       x = array_ops.placeholder(dtype, shape=shape, name=name)
    return x


def cross_args(args):
    print(type(args))
    if not isinstance(args, list) and not isinstance(args, tuple):
        raise ValueError('Expected a list of ltn tensors.')

    for tensor in args:
        if not is_ltn_tensor(tensor):
            raise ValueError('Unexpected tensor. Expected a list of ltn tensors.')

    result = args[0]

    for arg in args[1:]:
        result, _ = cross_2args(result, arg)

    result_flat = tf.reshape(result,
                             (tf.math.reduce_prod(tf.shape(result)[:-1]),
                              tf.shape(result)[-1]))

    result_args = tf.split(result_flat, [tf.shape(arg)[-1] for arg in args], 1)
    return result, result_args


def cross_2args(X, Y):
    if X._doms == [] and Y._doms == []:
        result = tf.concat([X, Y], axis=-1)
        result._doms = []
        return result, [X, Y]

    X_Y = set(X._doms) - set(Y._doms)
    Y_X = set(Y._doms) - set(X._doms)

    eX = X
    eX_doms = [x for x in X._doms]
    for y in Y_X:
        eX = tf.expand_dims(eX, 0)
        eX_doms = [y] + eX_doms

    eY = Y
    eY_doms = [y for y in Y._doms]
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

    result1._doms = eX_doms
    result2._doms = eX_doms
    result._doms = eX_doms

    return result, [result1, result2]
