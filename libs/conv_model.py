from tensorflow import keras
from Brett import prepare_data


def conv_layer_args(H):
    
    args = {
        'filters': H['filter_count'], 
        'padding': 'same', 
        'activation': H['activation']
    }

    args_a = {
        'kernel_size': H['kernel_size_a'], 
        'strides': H['stride_a'],
        'use_bias': False,
        **args
    }

    args_b = {'kernel_size': H['kernel_size_b'], **args}
    
    return args_a, args_b


def build(H):

    args_a, args_b = conv_layer_args(H)
    
    m = len(prepare_data.INPUT_SIGS)
    
    x = keras.layers.Input(shape=(H['window_size'], m))
    
    Z = []
    for i in range(m):
        z = x[:, :, i:i+1]
        for j in range(H['layer_count_a']):
            z = keras.layers.Conv1D(**args_a)(z)
        Z.append(z)

    z = keras.layers.Concatenate()(Z)

    for i in range(H['layer_count_b']):
        if i > 0:
            Z = [keras.layers.Conv1D(**args_b)(z), z[:, :, :H['filter_count']]]
            z = keras.layers.Concatenate()(Z)
        else:
            z = keras.layers.Conv1D(**args_b)(z)

    z = keras.layers.Flatten()(z)
    z = keras.layers.Dense(H['dense_units'], activation=H['activation'])(z)
    if H['dropout'] > 0:
        z = keras.layers.Dropout(H['dropout'])(z)
    z = keras.layers.Dense(2)(z)

    model = keras.models.Model(inputs=x, outputs=z)

    optimizer = getattr(keras.optimizers, H['optimizer']['name'].title())

    model.compile(
        optimizer = optimizer(**H['optimizer']['args']),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model