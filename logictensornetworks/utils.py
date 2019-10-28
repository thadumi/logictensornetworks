import tensorflow as tf


def predicateDefinition(number_of_features_or_vars, layers=None, label='predicate'):
    layers = layers or 4
    if type(number_of_features_or_vars) is list:  # list of vars I suppose
        number_of_features = sum([int(v.shape[1]) for v in number_of_features_or_vars])
    elif type(number_of_features_or_vars) is tf.Tensor:
        number_of_features = int(number_of_features_or_vars.shape[1])
    else:
        number_of_features = number_of_features_or_vars

    # if there is not a custom predicate model defined create create the default schema
    w = tf.Variable(
        tf.random.normal(
            [layers,
             number_of_features + 1,
             number_of_features + 1], mean=0, stddev=1), name="W" + label)

    W = tf.linalg.band_part(w, 0, -1)
    u = tf.Variable(tf.ones([layers, 1]),
                    name="u" + label)

    def pred_definition(*args):
        app_label = (label + "/" + "_".join([arg.name.split(":")[0] for arg in args]) + "/") \
            if not tf.executing_eagerly() else (label + '_applied_pred')

        tensor_args = tf.concat(args, axis=1)
        X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                       tensor_args], 1)
        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [layers, 1, 1]), W)
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])
        gX = tf.matmul(tf.tanh(XWX), u)
        result = tf.sigmoid(gX, name=app_label)
        return result

    pred_definition.vars = [w, W, u]
    return pred_definition
