from tensorflow.compat.v1.keras import layers


def concat_dense(inputs, layer_sizes, activation="relu", kernel_regularizer=None, dropout_rate=0.1):
    outputs = []
    x = inputs
    for size in layer_sizes:
        x = layers.Dense(size, activation=activation, kernel_regularizer=kernel_regularizer)(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs.append(x)
    x = layers.Concatenate()(outputs)
    return x
