import tensorflow as tf
from tensorflow import keras as K
from functools import partial, reduce
import numpy
import loss_metrics

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)


def resnet_block_a(H, x):
    
    mask = ~tf.reduce_all(x == 0, axis=[1, 2], keepdims=True)
    mask = tf.cast(mask, dtype=x.dtype)
    
    conv_args = {
        'filters': H['filter_count'], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_size': H['kernel_sizes'][1], 
        'strides': H['strides'][1],   
    }
    
    layers = [
        K.layers.BatchNormalization(),
        partial(tf.multiply, mask),
        K.layers.Activation('relu'),
        K.layers.Conv1D(**conv_args),
        K.layers.BatchNormalization(),
        partial(tf.multiply, mask),
        K.layers.Activation('relu'),
        K.layers.Conv1D(**conv_args),
    ]
    
    f = compose(*layers[::-1])
    z = f(x) + x[:, ::H['strides'][1]**2]
    return z


def resnet_block_b(H):
    conv_args = {
        'filters': H['filter_count'], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_size': H['kernel_sizes'][2],
    }
    
    layers = [
        K.layers.BatchNormalization(),
        K.layers.Activation('relu'),
        K.layers.Conv1D(**conv_args),
        K.layers.BatchNormalization(),
        K.layers.Activation('relu'),
        K.layers.Conv1D(**conv_args),
    ]
    
    f = compose(*layers[::-1])
    block = lambda x: f(x) + x
    return block
    

def resnet_block_ab(H, mask):
    
    conv_args = {
        'filters': H['filter_count'], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
    }
    
    conv_args_a = {
        'kernel_size': (H['kernel_sizes'][2], len(H['input_sigs_train'])), 
        'strides': (1, len(H['input_sigs_train'])),
        **conv_args
    }

    conv_args_b = {
        'kernel_size': H['kernel_sizes'][2],
        **conv_args
    }
    
    layers = [
        K.layers.BatchNormalization(),
        partial(tf.multiply, tf.expand_dims(mask, axis=-1)),
        K.layers.Activation('relu'),
        K.layers.Conv2D(**conv_args_a),
        partial(tf.squeeze, axis=2),
        K.layers.BatchNormalization(),
        K.layers.Activation('relu'),
        K.layers.Conv1D(**conv_args_b)
    ]
    
    f = compose(*layers[::-1])
    block = lambda x: f(x) + tf.reduce_sum(x, axis=2)
    return block


def first_layer(H, mask):
    args = {
        'filters': H['filter_count'],
        'padding': 'same',
        'activation': 'relu',
        'kernel_size': (H['kernel_sizes'][0], 1), 
        'strides': (H['strides'][0], 1),
    }
    layers = [
        partial(tf.expand_dims, axis=-1),
        K.layers.Conv2D(**args),
        partial(tf.multiply, tf.expand_dims(mask, axis=-1))
    ]
    layer = compose(*layers[::-1])
    return layer


def dense_layer(H):
    layers = [
        K.layers.Flatten(),
        K.layers.Dense(H['dense_units'], activation='relu'),
    ]
    
    if H['dropout'] > 0:
        layers.append(K.layers.Dropout(H['dropout']))
    
    layer = compose(*layers[::-1])
    return layer


def final_layer(H):
    layers = [
        K.layers.Dense(2 * len(H['output_sigs'])),
        K.layers.Reshape([2, len(H['output_sigs'])])
    ]
    layer = compose(*layers[::-1])
    return layer


def build(H):

    m = len(H['input_sigs_train'])
    x = K.layers.Input(shape=(H['window_size'], m), batch_size=H['batch_size'])
    
    mask = ~tf.reduce_all(x == 0, axis=1, keepdims=True)
    mask = tf.cast(mask, dtype=x.dtype)
    
    pressure_layer = K.layers.Dense(2 * len(H['output_sigs']))
    
    split = partial(tf.split, axis=2, num_or_size_splits=m)
    squeeze = partial(map, partial(tf.squeeze, axis=2))
    stack = partial(tf.stack, axis=2)
        
    layers = [
        first_layer(H, mask),
        split, squeeze,
        partial(map, partial(resnet_block_a, H)), 
        list, stack,
        resnet_block_ab(H, mask),
        *[resnet_block_b(H) for i in range(H['layer_count'])],
        dense_layer(H),
    ]
    
    net = compose(*layers[::-1])
    z = net(x)
    
    reshape = K.layers.Reshape([2, len(H['output_sigs'])], name='pressure')
    pressure = compose(reshape, pressure_layer)(z)
    gender = K.layers.Dense(1, name='gender', activation='sigmoid')(z)
    died = K.layers.Dense(1, name='died', activation='sigmoid')(z)
    
    model = K.models.Model(inputs=x, outputs=[pressure, gender, died])
    
    mean_pressure = [H['mean_pressure'][s] for s in H['output_sigs']]
    mean_pressure = numpy.array(mean_pressure, dtype='float32').T
                                
    pressure_layer.set_weights([
        pressure_layer.get_weights()[0],
        tf.constant(mean_pressure.flatten())
    ])  

    lr_schedule = K.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = [H['steps_per_epoch'] * i for i in H['lr_boundaries']],
        values = [H['learning_rate'] / i for i in H['lr_divisors']]
    )
    
    loss, metrics = loss_metrics.build(H)
    
    loss_weights = H['loss_weights']['output']
    
    model.compile(
        optimizer = K.optimizers.Adam(learning_rate=lr_schedule),
        loss = loss,
        metrics = metrics,
        loss_weights = {
            'pressure': loss_weights['pressure'],
            'gender': loss_weights['gender'],
            'died': loss_weights['died']
        },
    )

    return model