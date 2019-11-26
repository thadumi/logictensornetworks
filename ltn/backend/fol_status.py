"""
:Date: Nov 26, 2019
:Version: 0.0.3
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
    pass


def train(max_epochs=10000,
          track_sat_levels=100,
          sat_level_epsilon=.99,
          optimizer=None):

    @tf.function
    def axioms_aggregator(axioms):
        return tf.reduce_mean(tf.concat(axioms, axis=0))

    # @tf.function
    def theory():
        axioms = [axiom.tensor for axiom in AXIOMS]
        return axioms_aggregator(axioms)

    def loss(x):
        return - (1.0 / x)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=.01, decay=.9)

    for step in range(max_epochs):
        # loss_step = loss(theory())
        optimizer.minimize(lambda: loss(theory()), var_list=lambda: _TF_VARIABLES)

