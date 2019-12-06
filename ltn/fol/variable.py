"""
:Author: Theodor A. Dumitrescu
:Date: Dec 05, 2019
:Version: 0.0.4
"""

import logging
from typing import List, Any

import tensorflow as tf

import ltn.fol.fol_status as FOL
from ltn.fol.constant import LogicalConstant
from ltn.fol.logic import LogicalComputation


class LogicalVariable(LogicalComputation):

    def __init__(self,
                 name: str = None,
                 static_value: Any = None,
                 constants: List[LogicalConstant] = None):
        super(LogicalVariable, self).__init__(None, (name,), ())
        self.name = name
        self.value = static_value

        self.closed_world = constants is not None
        self.constants = constants

        if self.closed_world:
            self._ltn_args = constants
        elif tf.is_tensor(static_value):  # static_value arg should have been provided
            self.value = tf.identity(static_value)
        else:  # static value is provided but is not a tensor
            self.value = tf.constant(static_value)

    def tensor(self) -> tf.Tensor:
        if self.closed_world:  # (thadumi): we could create two subclass ClosedWordVariable and StaticVariable
            return self._compute(*[constant.tensor() for constant in self.constants])
        else:
            return self.value

    @tf.function
    def _compute(self, *args):
        logging.debug('tracking variable {}'.format(self.name))
        # only for closed word variables
        return tf.concat(args, axis=0)

    def __str__(self):
        return self.name


def variable(name: str, tensor: tf.Tensor = None, constants: List[LogicalConstant] = None) -> LogicalVariable:
    if FOL.variable_already_defined(name):
        msg = '[variable] There is already a variable having the name `{}`'.format(name)
        logging.error(msg)

        raise ValueError(msg)

    var = LogicalVariable(name=name, static_value=tensor, constants=constants)
    FOL.track_variable(name, var)

    return var
