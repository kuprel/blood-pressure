from tensorflow import keras
import tensorflow as tf

def resnet_block_a(x, args):
    args['padding'] = 'same'
    f1 = keras.layers.Conv2D(**args, activation='relu')
    f2 = keras.layers.Conv2D(**args, activation=None)
    relu = keras.layers.Activation('relu')
    z = relu(f2(f1(x)) + x[:, ::args['strides'][0]**2])
    return z

def resnet_block_b(x, args):
    args['padding'] = 'same'
    f1 = keras.layers.Conv1D(**args, activation='relu')
    f2 = keras.layers.Conv1D(**args, activation=None)
    relu = keras.layers.Activation('relu')
    z = relu(f2(f1(x)) + x)
    return z

def resnet_block_ab(x, args1, args2):
    args1['padding'] = args2['padding'] = 'same'
    f1 = keras.layers.Conv2D(**args1, activation='relu')
    f2 = keras.layers.Conv1D(**args2, activation=None)
    relu = keras.layers.Activation('relu')
    z = f2(tf.squeeze(f1(x), axis=2))
    z = relu(z + tf.reduce_sum(x, axis=2))
    return z


def build(H):

    m = len(H['input_sigs_train'])
    x = keras.layers.Input(shape=(H['window_size'], m))

    args = {
        'filters': H['filter_count'], 
        'kernel_size': (H['kernel_sizes'][0], 1), 
        'strides': (H['strides'][0], 1),
        'use_bias': False,
        'padding': 'same',
        'activation': 'relu'
    }

    z = tf.expand_dims(x, axis=-1)
    layer = keras.layers.Conv2D(**args)
    z = layer(z)

    args = {
        'filters': H['filter_count'], 
        'kernel_size': (H['kernel_sizes'][1], 1), 
        'strides': (H['strides'][1], 1),
        'use_bias': False,
    }
    z = resnet_block_a(z, args)

    args1 = {
        'filters': H['filter_count'], 
        'kernel_size': (H['kernel_sizes'][2], m), 
        'strides': (1, m),
    }

    args2 = {
        'filters': H['filter_count'], 
        'kernel_size': H['kernel_sizes'][2],
    }

    z = resnet_block_ab(z, args1, args2)

    args = {
        'filters': H['filter_count'], 
        'kernel_size': H['kernel_sizes'][2],
    }

    for i in range(H['layer_count']):
        z = resnet_block_b(z, args)

    z = keras.layers.Flatten()(z)
    z = keras.layers.Dense(H['dense_units'], activation='relu')(z)
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