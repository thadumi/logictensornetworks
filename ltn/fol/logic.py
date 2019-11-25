"""
:Author: thadumi
:Date: 25/11/19
:Version: 0.0.2
"""

import tensorflow as tf

from ltn_utils import cross_args, cross_2args


class LogicalComputation(object):
    def __init__(self,
                 in_doms,
                 out_doms,
                 args):
        self._in_doms = in_doms  # TODO(thadumi): this could be removed and computed using _ltn_args
        self._out_doms = out_doms
        self._ltn_args = args

    @property
    def doms(self):
        return self._out_doms

    @property
    def tensor(self) -> tf.Tensor:
        # call the definition aka definition(*[arg.tensor() for arg in args])
        pass

    @property
    def numpy(self):
        return self.tensor.numpy()

    def and_(self, other):
        return self.__and__(other)

    def _and_(self, other):
        return self.__and__(other)

    def __and__(self, other):
        print(str(self) + ' and ' + str(other))
        tensor_cross_args, result_doms = cross_args([self._out_doms, other.doms])
        return AndLogicalComputation(in_doms=[self.doms, other.doms], out_doms=result_doms, args=[self, other])

    def or_(self, other):
        return self.__or__(other)

    def _or_(self, other):
        return self.__or__(other)

    def __or__(self, other):
        tensor_cross_args, result_doms = cross_args([self._out_doms, other.doms])
        return OrLogicalComputation(in_doms=[self.doms, other.doms], out_doms=result_doms, args=[self, other])

    def not_(self):
        return self.__invert__()

    def __invert__(self):
        return NotLogicalComputation(self.doms, self.doms, self)

    def __rshift__(self, other):
        eX_doms, _, _, tensor_cross_2args = cross_2args(x_ltn_doms=self.doms, y_ltn_doms=other.doms)
        return ImpliesLogicalComputation(in_doms=[self.doms, other.doms], out_doms=eX_doms, args=[self, other])

    def __eq__(self, other):
        # WARN overriding __eq__ could cause issues on dictionaries
        eX_doms, _, _, tensor_cross_2args = cross_2args(x_ltn_doms=self.doms, y_ltn_doms=other.doms)
        return EquivalenceLogicalComputation(in_doms=[self.doms, other.doms], out_doms=eX_doms, args=[self, other])


# TODO(thadumi): AND should be a monadic type and return itself during an AND operation
# ie AndLC.and(LC) -> AndLC has a new argument which is LC
# the same of OR
# this could be done via __rand__ (the operations are aggregated from lest to right)

class AndLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args=None):
        super(AndLogicalComputation, self).__init__(in_doms, out_doms, args)

    def __str__(self):
        return ' ∧ '.join([str(arg) for arg in self._ltn_args])


class OrLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args=None):
        super(OrLogicalComputation, self).__init__(in_doms, out_doms, args)

    def __str__(self):
        return ' ∨ '.join([str(arg) for arg in self._ltn_args])


class NotLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args):
        super(NotLogicalComputation, self).__init__(in_doms, out_doms, args)

    def __str__(self):
        return '¬' + str(self._ltn_args)


class ImpliesLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args):
        super(ImpliesLogicalComputation, self).__init__(in_doms, out_doms, args)

    def __str__(self):
        return str(self._ltn_args[0]) + ' ⇒ ' + str(self._ltn_args[1])


class EquivalenceLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args):
        super(EquivalenceLogicalComputation, self).__init__(in_doms, out_doms, args)

    def __str__(self):
        return str(self._ltn_args[0]) + ' ⇔ ' + str(self._ltn_args[1])


class ForAllLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, variables, proposition):
        super(ForAllLogicalComputation, self).__init__(in_doms, out_doms, args=[*variables, proposition])
        self.vars = variables
        self.proposition = proposition

    def __str__(self):
        return '∀ ' + ','.join([str(var) for var in self.vars]) + ': ' + str(self.proposition)


class ExistsLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, variables, proposition):
        super(ExistsLogicalComputation, self).__init__(in_doms, out_doms, args=[*variables, proposition])
        self.vars = variables
        self.proposition = proposition

    def __str__(self):
        return '∃ ' + ','.join([str(var) for var in self.vars]) + ': ' + str(self.proposition)


def Not(lc: LogicalComputation) -> LogicalComputation:
    return lc.not_()


def Forall(variables, proposition: LogicalComputation):
    # TODO(thadumi) assert variables and proposition type

    if type(variables) is not list or type(variables) is not tuple:
        variables = (variables,)

    result_doms = [x for x in proposition.doms if x not in [var.doms[0] for var in variables]]
    return ForAllLogicalComputation([*[var.doms for var in variables], proposition.doms], result_doms,
                                    variables, proposition)


def Exist(variables, proposition: LogicalComputation):
    if type(variables) is not list or type(variables) is not tuple:
        variables = (variables,)

    result_doms = [x for x in proposition.doms if x not in [var.doms[0] for var in variables]]
    return ExistsLogicalComputation([*[var.doms for var in variables], proposition.doms], result_doms,
                                    variables, proposition)
