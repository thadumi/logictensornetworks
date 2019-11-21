import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import logictensornetworks as ltn
from logictensornetworks import Not, Implies, Forall, Exists, Equiv, And

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format


def plt_heatmap(df):
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.colorbar()


pd.set_option('precision', 2)

ltn.LAYERS = 4
ltn.BIAS_factor = 1e-7
ltn.set_universal_aggregator("mean")

size = 20
g1 = {l: ltn.constant(label=l, min_value=[0.] * size, max_value=[1.] * size) for l in 'abcdefgh'}
g2 = {l: ltn.constant(label=l, min_value=[0.] * size, max_value=[1.] * size) for l in 'ijklmn'}
g = {**g1, **g2}

friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
           ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]
smokes = ['a', 'e', 'f', 'g', 'j', 'n']
cancer = ['a', 'e']

p = ltn.variable(tf.concat(list(g.values()), axis=0), label='p')

q = ltn.variable(tf.concat(list(g.values()), axis=0), label='q')

p1 = ltn.variable(tf.concat(list(g1.values()), axis=0), label='p1')

q1 = ltn.variable(tf.concat(list(g1.values()), axis=0), label='q1')

p2 = ltn.variable(tf.concat(list(g2.values()), axis=0), label='p2')

q2 = ltn.variable(tf.concat(list(g2.values()), axis=0), label='q2')

Friends = ltn.predicate(size * 2, label='Friends')
Smokes = ltn.predicate(size, label='Smokes')
Cancer = ltn.predicate(size, label='Cancer')

variables = list(g.values())
# variables.append(Friends.vars[0])
variables.append(Friends.vars[1])
# variables.append(Smokes.vars[0])
variables.append(Smokes.vars[1])
# variables.append(Cancer.vars[0])
variables.append(Cancer.vars[1])

'''
# @tf.function
def fiends():
    return tf.concat([Friends(g[x], g[y]) for (x, y) in friends], axis=0)


# @tf.function
def not_fiends():
    return tf.concat([Not(Friends(g[x], g[y])) for x in g1 for y in g1
                      if (x, y) not in friends and x < y] +
                     [Not(Friends(g[x], g[y])) for x in g2 for y in g2
                      if (x, y) not in friends and x < y]
                     , axis=0)


# @tf.function
def smokers():
    return tf.concat([Smokes(g[x]) for x in smokes], axis=0)
'''

d = And(g['a'], g['b'])
print(tf.shape(d))
print(tf.shape(d))
print(d)


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
    return -(1.0 / tf.reduce_mean(1 / x)) + ltn.BIAS


optimizer = tf.keras.optimizers.RMSprop(learning_rate=.01, decay=.9)

# Iterate over the batches of the dataset.
for step in range(100):
    optimizer.minimize(lambda: loss(theory()), var_list=lambda: variables)
    # Log every 200 batches.
    if step % 10 == 0:
        print(step, "=====>", ltn.getBias().numpy())

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
