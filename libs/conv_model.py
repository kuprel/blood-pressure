import tensorflow as tf
from tensorflow import keras as K
from functools import partial, reduce
import numpy
import loss_metrics

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)


def resnet_block_a(H, mask, x):
    
#     mask = ~tf.reduce_all(x == 0, axis=[1, 2], keepdims=True)
#     mask = tf.cast(mask, dtype=x.dtype)
    
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
        'kernel_size': (H['kernel_sizes'][2], len(H['input_sigs'])), 
        'strides': (1, len(H['input_sigs'])),
        **conv_args
    }

    conv_args_b = {
        'kernel_size': H['kernel_sizes'][2],
        **conv_args
    }
    
    layers = [
        K.layers.BatchNormalization(),
        partial(tf.multiply, mask),
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
        partial(tf.multiply, mask)
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


def build(H, diagnosis_codes):

    input_shape = (H['window_size'], len(H['input_sigs']))
    signals = K.layers.Input(
        shape = input_shape, 
        batch_size = H['batch_size'],
        name = 'signals'
    )
    
#     mask = ~tf.reduce_all(signals == 0, axis=1, keepdims=True)

    mask = K.layers.Input(
        shape = len(H['input_sigs']),
        batch_size = H['batch_size'],
        name = 'mask'
    )
        
    signals_mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=-1)
    signals_mask = tf.cast(signals_mask, dtype=signals.dtype)
    
    pressure_layer = K.layers.Dense(2 * len(H['output_sigs']))
    
    split = partial(tf.split, axis=2, num_or_size_splits=len(H['input_sigs']))
    squeeze = partial(map, partial(tf.squeeze, axis=2))
    stack = partial(tf.stack, axis=2)
        
    layers = [
        first_layer(H, signals_mask),
        split, squeeze,
        partial(map, partial(resnet_block_a, H, signals_mask)), 
        list, stack,
        resnet_block_ab(H, signals_mask),
        *[resnet_block_b(H) for i in range(H['layer_count'])],
        dense_layer(H),
    ]
    
    net = compose(*layers[::-1])
    z = net(signals)
    
    reshape = K.layers.Reshape([2, len(H['output_sigs'])], name='pressure')
    pressure = compose(reshape, pressure_layer)(z)
    
    diagnosis = K.layers.Dense(
        len(diagnosis_codes), 
        name='diagnosis', 
        activation='sigmoid'
    )(z)
        
    model = K.models.Model(
        inputs=[signals, mask], 
        outputs=[pressure, diagnosis]
    )
    
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
    
    loss, metrics = loss_metrics.build(H, diagnosis_codes)
    
    loss_weights = H['loss_weights']['output']
    
    model.compile(
        optimizer = K.optimizers.Adam(learning_rate=lr_schedule),
        loss = loss,
        metrics = metrics,
        loss_weights = {k: loss_weights[k] for k in ['pressure', 'diagnosis']}
    )

    return model