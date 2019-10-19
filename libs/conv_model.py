from tensorflow import keras


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
    
    m = len(H['input_sigs_train'])
    
    x = keras.layers.Input(shape=(H['window_size'], m))
    
    Z = [x[:, ::H['input_stride'], j:j+1] for j in range(m)]
    
    for i in range(H['layer_count_a']):
        for j in range(m):
            layer = keras.layers.Conv1D(**args_a)
            z = [layer(Z[j]), Z[j][:, ::H['stride_a'], :H['filter_count']]]
            Z[j] = keras.layers.Concatenate()(z)

    z = keras.layers.Concatenate()(Z)

    for i in range(H['layer_count_b']):
        layer = keras.layers.Conv1D(**args_b)
        if i > 0:
            z = [layer(z), z[:, :, :H['filter_count']]]
            z = keras.layers.Concatenate()(z)
        else:
            z = layer(z)

    z = keras.layers.Flatten()(z)
    z = keras.layers.Dense(H['dense_units'], activation=H['activation'])(z)
    if H['dropout'] > 0:
        z = keras.layers.Dropout(H['dropout'])(z)
        
    final_layer = keras.layers.Dense(2)
    z = final_layer(z)

    model = keras.models.Model(inputs=x, outputs=z)

    final_layer.set_weights([
        final_layer.get_weights()[0],
        keras.backend.constant([120, 60], dtype='float32')
    ])
    
    optimizer = getattr(keras.optimizers, H['optimizer']['name'].title())

    model.compile(
        optimizer = optimizer(**H['optimizer']['args']),
        loss='mean_absolute_error'
#         loss='mean_squared_error',
#         metrics=['mean_absolute_error']
    )
    
    return model