"""
:Date: Nov 16, 2019
:Version: 0.0.1
"""
from typing import List, Dict, Any

import tensorflow as tf


CONSTANTS = {}
PREDICATES = {}
VARIABLES = {}  # TODO(thadumi)
FUNCTIONS = {}  # NOTE(thadumi): useless a function is a predicate with one arg
TERMS = {}  # NOTE(thadumi): useless
FORMULAS = {}  # NOTE(thadumi): useless

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


def variable(inital_value):
    var = tf.Variable(inital_value)
    _TF_VARIABLES.append(var)

    return var


def axiom(lc):
    # TODO(thadumi): an axiom can't be a constant or a variable. Add checks
    AXIOMS.append(lc)
