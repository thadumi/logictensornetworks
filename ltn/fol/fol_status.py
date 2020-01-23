"""
:Date: Dec 16, 2019
:Version: 0.0.5
"""
import logging
import time
from typing import List, Any, Callable

import tensorflow as tf

CONSTANTS = {}
PREDICATES = {}
VARIABLES = {}  # TODO(thadumi)
FUNCTIONS = {}

# TODO(thadumi) define an hash method for LogicalComputation for checking that there are no others axiom already defined
AXIOMS = []

_TF_VARIABLES: List[tf.Tensor] = []  # all the variables defined by the FOL operators

_LOGICAL_MODE = True  # TODO(thadumi): for next version eager mode (compute on the fly the tensors)


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


def track_function(name, meta):
    FUNCTIONS[name] = meta


def function_already_defined(name: str) -> bool:
    return name in FUNCTIONS.keys()


def track_variable(name: str, meta: Any):
    VARIABLES[name] = meta


def variable_already_defined(name: str) -> bool:
    return name in VARIABLES.keys()


def variable(initial_value, **kwargs):
    var = tf.Variable(initial_value, **kwargs)
    _TF_VARIABLES.append(var)

    return var


def tell(lc):
    AXIOMS.append(lc)


def ask(lc):
    return lc.numpy()


def train(max_epochs: int = 10000,
          optimizer: tf.keras.optimizers.Optimizer = None,  # NOTE(thadumi): define covariance
          loss_function: Callable[[tf.Tensor], tf.Tensor] = None,
          axioms_aggregator: Callable[..., tf.Tensor] = None,
          track_sat_levels: int = 100):
    if axioms_aggregator is None:
        def default_axioms_aggregator(*axioms):
            return 1 / tf.concat(axioms, axis=0)

        axioms_aggregator = default_axioms_aggregator

    @tf.function
    def theory():
        return axioms_aggregator(*[axiom.tensor() for axiom in AXIOMS])

    if loss_function is None:
        @tf.function
        def default_loss_function(x):
            return 0.0 - (1.0 / tf.math.reduce_mean(x))

        loss_function = default_loss_function

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

    start_time = time.time_ns()
    for step in range(max_epochs):
        # TODO(thadumi) processing gradients before applying them for tracking the loss history without computing it
        #  twice
        optimizer.minimize(lambda: loss_function(theory()), var_list=lambda: _TF_VARIABLES)

        if step % track_sat_levels == 0:
            current_time = time.time_ns()
            delta_time = (current_time - start_time) // 1000000
            loss_step = loss_function(theory())
            logging.info('Step {:10}\t\tLoss {:25}\t in {}ms'.format(step, loss_step, delta_time))
            start_time = time.time_ns()
