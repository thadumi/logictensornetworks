"""
:Author: thadumi
:Date: 26/11/19
:Version: 0.0.5
"""

import tensorflow as tf

from ltn.backend.ltn_utils import cross_args, cross_2args, split_cross_2args
from ltn.backend.norms import F_And, F_Or, F_Not, F_Implies, F_Equiv, F_Exists, F_ForAll


class LogicalComputation(object):
    def __init__(self,
                 in_doms,
                 out_doms,
                 args):
        if type(in_doms) is not tuple and in_doms is not None:
            in_doms = tuple(in_doms)

        if type(out_doms) is not tuple:
            out_doms = tuple(out_doms)

        self._in_doms = in_doms  # TODO(thadumi): this could be removed and computed using _ltn_args
        self._out_doms = out_doms
        self._ltn_args = args or ()

    @property
    def doms(self):
        return self._out_doms

    # @property
    def tensor(self) -> tf.Tensor:
        # call the definition aka definition(*[arg.tensor() for arg in args])
        # raise Exception(self.__class__.__name__ + ' needs to define the tensor method')
        args = [arg.tensor() for arg in self._ltn_args]
        return self._compute(*args)

    def numpy(self):
        if tf.executing_eagerly():
            return self.tensor().numpy()
        else:
            raise Exception('Operation not supported in eager mode')

    def _compute(self, *args):
        raise Exception(self.__class__.__name__ + ' needs to define the _compute method')

    def and_(self, other):
        return self.__and__(other)

    def _and_(self, other):
        return self.__and__(other)

    def __and__(self, other):
        tensor_cross_args, result_doms = cross_args(self._out_doms, other.doms)
        return AndLogicalComputation(in_doms=(self.doms, other.doms),
                                     out_doms=result_doms,
                                     args=(self, other),
                                     tensor_cross_args=tensor_cross_args)

    def or_(self, other):
        return self.__or__(other)

    def _or_(self, other):
        return self.__or__(other)

    def __or__(self, other):
        tensor_cross_args, result_doms = cross_args(self._out_doms, other.doms)
        return OrLogicalComputation(in_doms=(self.doms, other.doms),
                                    out_doms=result_doms,
                                    args=(self, other),
                                    tensor_cross_args=tensor_cross_args)

    def negated(self):
        return self.__invert__()

    def not_(self):
        return self.__invert__()

    def __invert__(self):
        return NotLogicalComputation(self.doms, self.doms, (self, ))

    def __rshift__(self, other):
        eX_doms, tensor_cross_2args = cross_2args(self.doms, other.doms)
        return ImpliesLogicalComputation(in_doms=(self.doms, other.doms),
                                         out_doms=eX_doms,
                                         args=(self, other),
                                         tensor_cross_2args=tensor_cross_2args)

    def __eq__(self, other):
        # WARN overriding __eq__ could cause issues on dictionaries
        eX_doms, tensor_cross_2args = cross_2args(self.doms, other.doms)
        return EquivalenceLogicalComputation(in_doms=(self.doms, other.doms),
                                             out_doms=eX_doms,
                                             args=(self, other),
                                             tensor_cross_2args=tensor_cross_2args)

    def __hash__(self):
        # needed by tf.function for hashing the self argument in logical predicate name
        # WARN maybe should be considered a better hash value (?)
        return hash(str(self))


# TODO(thadumi): AND should be a monadic type and return itself during an AND operation
# ie AndLC.and(LC) -> AndLC has a new argument which is LC
# the same of OR
# this could be done via __rand__ (the operations are aggregated from lest to right)

class AndLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args, tensor_cross_args):
        super(AndLogicalComputation, self).__init__(in_doms, out_doms, args)
        self.tensor_cross_args = tensor_cross_args

    @tf.function
    def _compute(self, *args):
        cross_wffs = self.tensor_cross_args(*args)
        return F_And(cross_wffs)

    def __str__(self):
        return ' ∧ '.join([str(arg) for arg in self._ltn_args])


class OrLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args, tensor_cross_args):
        super(OrLogicalComputation, self).__init__(in_doms, out_doms, args)
        self.tensor_cross_args = tensor_cross_args

    @tf.function
    def _compute(self, *args):
        cross_wffs, _ = self.tensor_cross_args(*args)
        return F_Or(cross_wffs)

    def __str__(self):
        return ' ∨ '.join([str(arg) for arg in self._ltn_args])


class NotLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args):
        super(NotLogicalComputation, self).__init__(in_doms, out_doms, args)

    @tf.function
    def _compute(self, *args):
        return F_Not(args[0])

    def __str__(self):
        return '¬' + str(self._ltn_args[0])


class ImpliesLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args, tensor_cross_2args):
        super(ImpliesLogicalComputation, self).__init__(in_doms, out_doms, args)
        self.tensor_cross_2args = tensor_cross_2args

    @tf.function
    def _compute(self, *args):
        cross_wffs = self.tensor_cross_2args(args[0], args[1])
        cross_wffs = split_cross_2args(cross_wffs)
        return F_Implies(cross_wffs[0], cross_wffs[1])

    def __str__(self):
        return str(self._ltn_args[0]) + ' ⇒ ' + str(self._ltn_args[1])


class EquivalenceLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, args, tensor_cross_2args):
        super(EquivalenceLogicalComputation, self).__init__(in_doms, out_doms, args)
        self.tensor_cross_2args = tensor_cross_2args

    @tf.function
    def _compute(self, *args):
        cross_wffs = self.tensor_cross_2args(args[0], args[1])
        cross_wffs = split_cross_2args(cross_wffs)
        return F_Equiv(cross_wffs[0], cross_wffs[1])

    def __str__(self):
        return str(self._ltn_args[0]) + ' ⇔ ' + str(self._ltn_args[1])


class ForAllLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, variables, proposition):
        super(ForAllLogicalComputation, self).__init__(in_doms, out_doms, args=[*variables, proposition])
        self.vars = variables
        self.proposition = proposition
        self.quantif_axis = [proposition.doms.index(var.doms[0]) for var in variables]

    @tf.function
    def _compute(self, *args):
        vars = args[:-1]
        wff = args[-1]

        not_empty_vars = tf.cast(tf.math.reduce_prod(tf.stack([tf.size(var) for var in vars])),
                                 dtype=tf.dtypes.bool)

        return tf.cond(pred=not_empty_vars,
                       true_fn=lambda: F_ForAll(self.quantif_axis, wff),
                       false_fn=lambda: tf.ones((1,) * (len(self._out_doms) + 1)))

    def __str__(self):
        return '∀ ' + ','.join([str(var) for var in self.vars]) + ': ' + str(self.proposition)


class ExistsLogicalComputation(LogicalComputation):
    def __init__(self, in_doms, out_doms, variables, proposition):
        super(ExistsLogicalComputation, self).__init__(in_doms, out_doms, args=[*variables, proposition])
        self.vars = variables
        self.proposition = proposition
        self.quantif_axis = [proposition.doms.index(var.doms[0]) for var in variables]

    def _compute(self, *args):
        # TODO(thadumi)
        vars = args[:-1]
        wff = args[-1]

        # not_empty_vars = tf.cast(tf.math.reduce_prod(tf.stack([tf.size(var) for var in vars])),
        #                         dtype=tf.dtypes.bool)
        # if not_empty_vars:
        return F_Exists(self.quantif_axis, wff)
        # else:
        #    return tf.ones((1,) * (len(self.doms) + 1))

    def __str__(self):
        return '∃ ' + ','.join([str(var) for var in self.vars]) + ': ' + str(self.proposition)


def Not(lc: LogicalComputation) -> LogicalComputation:
    return lc.negated()


def And(arg1: LogicalComputation, arg2: LogicalComputation) -> LogicalComputation:
    return arg1.and_(arg2)


def Or(arg1: LogicalComputation, arg2: LogicalComputation) -> LogicalComputation:
    return arg1.or_(arg2)


def Implies(arg1, arg2):
    return arg1 >> arg2


def Equiv(arg1, arg2):
    return arg1 == arg2


def Forall(variables, proposition: LogicalComputation):
    # TODO(thadumi) assert variables and proposition type
    # TODO(thadumi) add doc for Forall functional API

    if type(variables) is not list and type(variables) is not tuple:
        variables = (variables,)

    result_doms = tuple([x for x in proposition.doms if x not in [var.doms[0] for var in variables]])

    return ForAllLogicalComputation([*[var.doms for var in variables], proposition.doms], result_doms,
                                    variables, proposition)


def Exists(variables, proposition):
    # TODO(thadumi) assert variables and proposition type
    # TODO(thadumi) add dock for Exist functional API

    if type(variables) is not list or type(variables) is not tuple:
        variables = (variables,)

    result_doms = [x for x in proposition.doms if x not in [var.doms[0] for var in variables]]
    return ExistsLogicalComputation([*[var.doms for var in variables], proposition.doms], result_doms,
                                    variables, proposition)
