"""
sample smokes_friends_cancer for testing the APIs

:Date: Nov 19, 2019
:Version: 0.0.1
"""

import tensorflow as tf

from constant import constant
from predicate import predicate
from variable import variable
from and_op import And
from or_op import Or
from not_op import Not
from equivalence_op import Equiv
from implies_op import Implies
from forall_op import Forall

size = 10

g1 = {l: constant(label=l, min_value=[0.] * size, max_value=[1.] * size) for l in 'abcdefgh'}
g2 = {l: constant(label=l, min_value=[0.] * size, max_value=[1.] * size) for l in 'ijklmn'}
g = {**g1, **g2}

friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
           ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]
smokes = ['a', 'e', 'f', 'g', 'j', 'n']
cancer = ['a', 'e']

Friends = predicate('Friends', size * 2)
Smokes = predicate('Smokes', size)
Cancer = predicate('Cancer', size)

p = variable('p', constants=list(g.values()))
q = variable('q', constants=list(g.values()))
p1 = variable('p1', constants=list(g1.values()))
q1 = variable('q1', constants=list(g1.values()))
p2 = variable('p2', constants=list(g2.values()))
q2 = variable('q2', constants=list(g2.values()))

def theory():
    facts = [Friends(g[x], g[y]) for (x, y) in friends] + \
            [Not(Friends(g[x], g[y])) for x in g1 for y in g1
             if (x, y) not in friends and x < y] + \
            [Not(Friends(g[x], g[y])) for x in g2 for y in g2
             if (x, y) not in friends and x < y] + \
            [Smokes(g[x]) for x in smokes] + \
            [Not(Smokes(g[x])) for x in g if x not in smokes] + \
            [Cancer(g[x]) for x in cancer] + \
            [Not(Cancer(g[x])) for x in g1 if x not in cancer] + \
            [Forall(p, Not(Friends(p, p))),
             Forall((p, q), Equiv(Friends(p, q), Friends(q, p))),
             Equiv(Forall(p1, Implies(Smokes(p1), Cancer(p1))),
                   Forall(p2, Implies(Smokes(p2), Cancer(p2)))),
             Equiv(Forall(p1, Implies(Cancer(p1), Smokes(p1))),
                   Forall(p2, Implies(Cancer(p2), Smokes(p2))))]

    return tf.concat(facts, axis=0)
