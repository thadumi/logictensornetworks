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


g1 = {l: const(min_value=[0.] * size, max_value=[1.] * size) for l in 'abcdefgh'}
