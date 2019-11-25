"""
:Author: Theodor A. Dumitrescu
:Date: 25/11/19
:Version: 0.0.1
"""

import logging
from typing import List

import tensorflow as tf

import ltn.backend.fol_status as FOL
from ltn.fol.constant import LogicalConstant
from ltn.fol.logic import LogicalComputation


class LogicalVariable(LogicalComputation):

    def __init__(self,
                 name=None,
                 static_value=None,
                 constants=None):
        super(LogicalVariable, self).__init__(None, [name])
        self.name = name
        self.value = static_value

        self.closed_world = constants is not None
        self.constants = constants

    def __str__(self):
        return self.name


def variable(name: str, tensor: tf.Tensor = None, constants: List[LogicalConstant] = None) -> LogicalVariable:
    if FOL.variable_already_defined(name):
        msg = '[variable] There is already a variable having the name `{}`'.format(name)
        logging.error(msg)

        raise ValueError(msg)

    if constants:
        constants = [c.name for c in constants]

    var = LogicalVariable(name=name, static_value=tensor, constants=constants)
    FOL.track_variable(name, var)

    return var
