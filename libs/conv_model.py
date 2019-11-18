import tensorflow as tf
from tensorflow import keras as K
from functools import partial, reduce
import numpy
import loss_metrics

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)


def resnet_block_a(H, mask, x):
    
    conv_args = {
        'filters': H['filter_count'], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_size': (H['kernel_sizes'][1], 1), 
        'strides': (H['strides'][1], 1),   
    }
    
    layers = [
        K.layers.BatchNormalization(),
        partial(tf.multiply, mask),
        K.layers.Activation('relu'),
        K.layers.Conv2D(**conv_args),
        K.layers.BatchNormalization(),
        partial(tf.multiply, mask),
        K.layers.Activation('relu'),
        K.layers.Conv2D(**conv_args),
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


def build(H, diagnosis_priors):

    input_shape = (H['window_size'], len(H['input_sigs']))
    signals = K.layers.Input(input_shape, name='signals')
    mask = K.layers.Input(input_shape[1], name = 'mask')
    
    float_mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=-1)
    float_mask = tf.cast(float_mask, dtype=signals.dtype)
    
    pressure_layer = K.layers.Dense(2 * len(H['output_sigs']))
    
    z = first_layer(H, float_mask)(signals)
    
    sig_groups = [slice(0, 1), slice(1, 2), slice(2, None)]
    Z = [z[:, :, G] for G in sig_groups]
    
    for i in range(H['layer_count_a']):
        for j, (G, z) in enumerate(zip(sig_groups, Z)):
            Z[j] = resnet_block_a(H, float_mask[:, :, G], z)
    
    norm = lambda w: w / tf.maximum(tf.reduce_sum(w, axis=2, keepdims=True), 1)
    weights = H['signal_feature_weight']
    weights = [weights['PLETH'], weights['RESP'], weights['ECG']]
    W = [norm(float_mask[:, :, G]) * w for G, w in zip(sig_groups, weights)]
    w = norm(tf.concat(W, axis=2))
    z = tf.concat(Z, axis=2)
    z = tf.reduce_sum(w * z, axis=2)

    for i in range(H['layer_count_b']):
        z = resnet_block_b(H)(z)

    features = dense_layer(H)(z)
    
    reshape = K.layers.Reshape([2, len(H['output_sigs'])], name='pressure')
    pressure = compose(reshape, pressure_layer)(features)
    
    diagnosis_layer = K.layers.Dense(
        len(diagnosis_priors), 
        name='diagnosis', 
        activation='sigmoid'
    )
    diagnosis = diagnosis_layer(features)
        
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
    
    diagnosis_layer.set_weights([
        diagnosis_layer.get_weights()[0],
        tf.math.log(diagnosis_priors) - tf.math.log(1 - diagnosis_priors)
    ])

    lr_schedule = K.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = [H['steps_per_epoch'] * i for i in H['lr_boundaries']],
        values = [H['learning_rate'] / i for i in H['lr_divisors']]
    )
    
    diagnosis_codes = list(diagnosis_priors.index)
    loss, metrics = loss_metrics.build(H, diagnosis_codes)
    
    loss_weights = H['loss_weights']['output']
    
    model.compile(
        optimizer = K.optimizers.Adam(learning_rate=lr_schedule),
        loss = loss,
        metrics = metrics,
        loss_weights = {k: loss_weights[k] for k in ['pressure', 'diagnosis']}
    )

    return model