"""
:Date: Dec 04, 2019
:Version: 0.0.4
"""
from typing import List, Any

import tensorflow as tf

CONSTANTS = {}
PREDICATES = {}
VARIABLES = {}  # TODO(thadumi)
FUNCTIONS = {}  # NOTE(thadumi): useless a function is a predicate with one arg

# TODO(thadumi) define an hash method for LogicalComputation for checking that there are no others axiom
# already defined
AXIOMS = []

_TF_VARIABLES: List[tf.Tensor] = []  # all the variables defined by the FOL operators

_LOGICAL_MODE = True  # TODO(thadumi): for next version eager mode (compute on the file the tensors)


def track_constant(constant_name, meta):
    """

    :param constant_name:
    :param meta:
    :return:
    """
    CONSTANTS[constant_name] = meta


def constant_already_defined(name: str) -> bool:
    return name in CONSTANTS.keys()


def track_predicate(predicate_name, meta):
    PREDICATES[predicate_name] = meta


def predicate_already_defined(name: str) -> bool:
    return name in PREDICATES.keys()


def track_variable(name: str, meta: Any):
    VARIABLES[name] = meta


def variable_already_defined(name: str) -> bool:
    return name in VARIABLES.keys()


def variable(initial_value, **kwargs):
    var = tf.Variable(initial_value, **kwargs)
    _TF_VARIABLES.append(var)

    return var


def tell(lc):
    # TODO(thadumi): an axiom can't be a constant or a variable. Add checks
    AXIOMS.append(lc)


def ask(lc):
    # TODO(thadumi)
    return lc.tensor()


def train(max_epochs=10000,
          track_sat_levels=100,
          sat_level_epsilon=.99,
          optimizer=None):

    def axioms_aggregator(*axioms):
        return 1 / tf.concat(axioms, axis=0)

    @tf.function
    def theory():
        return axioms_aggregator(*[axiom.tensor() for axiom in AXIOMS])

    @tf.function
    def loss(x):
        return 0.0 - (1.0 / tf.math.reduce_mean(x))

    optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

    # TODO(thadumi) processing gradients before applying them for tracking the loss history
    for step in range(max_epochs):
        optimizer.minimize(lambda: loss(theory()), var_list=lambda: _TF_VARIABLES)

        if step % 100 == 0:
            loss_step = loss(theory())
            print('Step {:10}\t\tLoss {}'.format(step, loss_step))
            # print(AXIOMS[0].numpy(), ' ', CONSTANTS['a'].numpy(), ' ', CONSTANTS['b'].numpy())
            # print(PREDICATES['Friends'].predicate_definition.w)
            # print(PREDICATES['Friends'].predicate_definition.u)
