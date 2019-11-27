"""
:Author: Theodor A. Dumitrescu
:Date: 26/11/19
:Version: 0.0.1
"""

import ltn.backend.fol_status as FOL
from constant import constant
from logic import Forall, Implies, Not, Exists, Equiv
from predicate import predicate
from variable import variable

size = 10

friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
           ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]
smokes = ['a', 'e', 'f', 'g', 'j', 'n']
cancer = ['a', 'e']

g1 = {l: constant(l, size=size) for l in 'abcdefgh'}
g2 = {l: constant(l, size=size) for l in 'ijklmn'}
g = {**g1, **g2}

Friends = predicate('Friends', 2, size)
Smokes = predicate('Smokes', 1, size)
Cancer = predicate('Cancer', 1, size)

p = variable('p', constants=list(g.values()))
q = variable('q', constants=list(g.values()))
p1 = variable('p1', constants=list(g1.values()))
q1 = variable('q1', constants=list(g1.values()))
p2 = variable('p2', constants=list(g2.values()))
q2 = variable('q2', constants=list(g2.values()))

for (x, y) in friends:
    FOL.tell(Friends(g[x], g[y]))

for x in g1:
    for y in g1:
        if (x, y) not in friends and x < y:
            FOL.tell(Friends(g[x], g[y]).not_())

for x in g2:
    for y in g2:
        if (x, y) not in friends and x < y:
            FOL.tell(Friends(g[x], g[y]).not_())

for x in smokes:
    FOL.tell(Smokes(g[x]))

for x in g:
    if x not in smokes:
        FOL.tell(Smokes(g[x]).not_())

for x in cancer:
    FOL.tell(Cancer(g[x]))

for x in g1:
    if x not in cancer:
        FOL.tell(Cancer(g[x]).not_())
FOL.tell(Forall(p, Not(Friends(p, p))))

FOL.tell(Forall((p, q), Equiv(Friends(p, q), Friends(q, p))))
FOL.tell(Equiv(Forall(p1, Implies(Smokes(p1), Cancer(p1))),
               Forall(p2, Implies(Smokes(p2), Cancer(p2)))))
FOL.tell(Equiv(Forall(p1, Implies(Cancer(p1), Smokes(p1))),
               Forall(p2, Implies(Cancer(p2), Smokes(p2)))))

FOL.train(max_epochs=1000)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


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

df_smokes_cancer = pd.DataFrame(tf.concat([Smokes(p).tensor(), Cancer(p).tensor()], axis=1).numpy(),
                                columns=["Smokes", "Cancer"],
                                index=list('abcdefghijklmn'))
pred_friends = tf.squeeze(Friends(p, q).tensor()).numpy()
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
print(Forall(p, Smokes(p) >> Forall(q, Friends(p, q) >> Smokes(q))).numpy())
