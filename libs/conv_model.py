from tensorflow import keras


def build(H):

    base_args = {'padding': 'same', 'activation': H['activation']}
    
    m = len(H['input_sigs_train'])
    
    x = keras.layers.Input(shape=(H['window_size'], m))
    
    Z = [x[:, ::H['input_stride'], j:j+1] for j in range(m)]
    
    for i, L in enumerate(H['layers_a']):
        args = {
            'filters': L['filter_count'], 
            'kernel_size': L['kernel_size'], 
            'strides': L['stride'],
            'use_bias': False,
            **base_args
        }
        if i == 0:
            args['name'] = 'conv_{}'.format(i)
            layer = keras.layers.Conv1D(**args)
        for j in range(m):
            if i != 0:
                args['name'] = 'conv_{}_{}'.format(i, j)
                layer = keras.layers.Conv1D(**args)
            z = [layer(Z[j]), Z[j][:, ::L['stride'], :L['filter_count']]]
            Z[j] = keras.layers.Concatenate()(z)

    z = keras.layers.Concatenate()(Z)

    args = {
        'filters': H['layers_b']['filter_count'], 
        'kernel_size': H['layers_b']['kernel_size'],
        **base_args
    }
    
    for i in range(H['layers_b']['count']):
        layer = keras.layers.Conv1D(**args)
        if i > 0:
            z = [layer(z), z[:, :, :H['layers_b']['filter_count']]]
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
        keras.backend.constant(H['mean_bp'], dtype='float32')
    ])    
    
    lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = [H['steps_per_epoch'] * i for i in H['lr_boundaries']],
        values = [H['learning_rate'] / i for i in H['lr_divisors']]
    )
    
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_absolute_error'
    )
    
    return model