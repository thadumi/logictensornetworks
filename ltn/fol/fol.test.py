"""
sample smokes_friends_cancer for testing the APIs

:Date: Nov 19, 2019
:Version: 0.0.1
"""

import tensorflow as tf

import ltn.backend.fol_status as FOL
from constant import constant
from equivalence_op import Equiv
from exists_op import Exists
from forall_op import Forall
from implies_op import Implies
from not_op import Not
from predicate import predicate
from variable import variable

print(len(FOL._TF_VARIABLES))

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


def loss(x):
    return - (1.0 / tf.reduce_mean(1 / x))


optimizer = tf.keras.optimizers.RMSprop(learning_rate=.01, decay=.9)

for step in range(300):
    # loss_step = loss(theory())

    optimizer.minimize(lambda: loss(theory()), var_list=lambda: FOL._TF_VARIABLES)

    if step % 10 == 0:
        print(step, "========>")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plt_heatmap(df):
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.colorbar()


pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format

pd.set_option('precision', 2)

df_smokes_cancer = pd.DataFrame(tf.concat([Smokes(p), Cancer(p)], axis=1).numpy(),
                                columns=["Smokes", "Cancer"],
                                index=list('abcdefghijklmn'))
pred_friends = tf.squeeze(Friends(p, q)).numpy()
df_friends_ah = pd.DataFrame(pred_friends[:8, :8],
                             index=list('abcdefgh'),
                             columns=list('abcdefgh'))
df_friends_in = pd.DataFrame(pred_friends[8:, 8:],
                             index=list('ijklmn'),
                             columns=list('ijklmn'))
plt.figure(figsize=(17, 5))
plt.subplot(131)
plt_heatmap(df_smokes_cancer)
plt.subplot(132)
plt_heatmap(df_friends_ah)
plt.subplot(133)
plt_heatmap(df_friends_in)
plt.show()

print("forall x ~Friends(x,x)",
      (Forall(p, Not(Friends(p, p)))).numpy())
print("Forall x Smokes(x) -> Cancer(x)",
      (Forall(p, Implies(Smokes(p), Cancer(p)))).numpy())
print("forall x y Friends(x,y) -> Friends(y,x)",
      (Forall((p, q), Implies(Friends(p, q), Friends(q, p)))).numpy())
print("forall x Exists y (Friends(x,y)",
      (Forall(p, Exists(q, Friends(p, q)))).numpy())
print("Forall x,y Friends(x,ay) -> (Smokes(x)->Smokes(y))",
      (Forall((p, q), Implies(Friends(p, q), Implies(Smokes(p), Smokes(q))))).numpy())
print("Forall x: smokes(x) -> forall y: friend(x,y) -> smokes(y))",
      (Forall(p, Implies(Smokes(p),
                         Forall(q, Implies(Friends(p, q),
                                           Smokes(q)))))).numpy())
