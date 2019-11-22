import tensorflow as tf
from tensorflow import keras as K
from functools import partial, reduce
import numpy
import loss_metrics

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)


MEAN_PRESSURE = {
    'ABP': [127, 58],
    'CVP': [18, 5.9],
    'ICP': [12.8, 6.4],
    'PAP': [43.2, 14.6]
}


def resnet_block_a(H, mask):
    
    conv_args = {
        'filters': 2 ** H['filter_count_log2'], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_size': (2 ** H['kernel_sizes_log2'][1], 1), 
        'strides': (2 ** H['strides_log2'][1], 1),
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
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
    block = lambda x: f(x) + x[:, ::(2**H['strides_log2'][1])**2]
    return block


def resnet_block_b(H):
    conv_args = {
        'filters': 2 ** H['filter_count_log2'], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_size': 2 ** H['kernel_sizes_log2'][2],
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
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
    

def resnet_block_ab(H, mask, group_count):

    conv_args = {
        'filters': 2 ** H['filter_count_log2'], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
    }

    conv_args_a = {
        'kernel_size': (2 ** H['kernel_sizes_log2'][2], group_count), 
        'strides': (1, group_count),
        **conv_args
    }

    conv_args_b = {
        'kernel_size': 2 ** H['kernel_sizes_log2'][2],
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
        'filters': 2 ** H['filter_count_log2'],
        'padding': 'same',
        'activation': 'relu',
        'kernel_size': (2 ** H['kernel_sizes_log2'][0], 1), 
        'strides': (2 ** H['strides_log2'][0], 1),
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
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
        K.layers.Dense(
            2 ** H['dense_units_log2'], 
            activation='relu',
            kernel_initializer = K.initializers.RandomNormal(
                mean=0.0, 
                stddev=2**H['initial_weights_std_log2']
            ),
        ),
    ]
    
    layers.append(K.layers.Dropout(2**H['dropout_log2']))
    
    layer = compose(*layers[::-1])
    return layer


def build(H, diagnosis_priors):

    input_shape = (2**H['window_size_log2'], len(H['input_sigs']))
    signals = K.layers.Input(input_shape, name='signals')
    mask = K.layers.Input(input_shape[1], name = 'mask')
    
    float_mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=-1)
    float_mask = tf.cast(float_mask, dtype=signals.dtype)
    
    pressure_layer = K.layers.Dense(2 * len(H['output_sigs']))
    
    z = first_layer(H, float_mask)(signals)
    
    sig_groups = [slice(0, 1), slice(1, 2), slice(2, None)]
    Z = [z[:, :, G] for G in sig_groups]
    
    for i in range(2**H['layer_count_a_log2']):
        for j, (G, z) in enumerate(zip(sig_groups, Z)):
            Z[j] = resnet_block_a(H, float_mask[:, :, G])(z)
       
    norm = lambda w: w / tf.maximum(tf.reduce_sum(w, axis=2, keepdims=True), 1)
    W = [norm(float_mask[:, :, G]) for G in sig_groups]
    Z = [tf.reduce_sum(w * z, axis=2, keepdims=True) for w, z in zip(W, Z)]
    z = tf.concat(Z, axis=2)
    
    group_mask = tf.concat([
        tf.reduce_any(float_mask[:, :, G] > 0, axis=2, keepdims=True) 
        for G in sig_groups
    ], axis=2)
    group_mask = tf.cast(group_mask, dtype=signals.dtype)
    
    z = resnet_block_ab(H, group_mask, group_count=len(sig_groups))(z)

    for i in range(2**H['layer_count_b_log2']):
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
    
    mean_pressure = [MEAN_PRESSURE[s] for s in H['output_sigs']]
    mean_pressure = numpy.array(mean_pressure, dtype='float32').T
                                
    pressure_layer.set_weights([
        pressure_layer.get_weights()[0],
        tf.constant(mean_pressure.flatten())
    ])
    
    if H['use_diagnosis_priors']:
        diagnosis_layer.set_weights([
            diagnosis_layer.get_weights()[0],
            tf.math.log(diagnosis_priors) - tf.math.log(1 - diagnosis_priors)
        ])

    boundaries = [2**H['steps_per_epoch_log2'] * i for i in H['lr_boundaries']]
    values = [2**(H['learning_rate_log2'] - i) for i in H['lr_divisors_log2']]
    lr_schedule = K.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = boundaries,
        values = values
    )
    
    diagnosis_codes = list(diagnosis_priors.index)
    loss, metrics = loss_metrics.build(H, diagnosis_codes)
    
    w = H['loss_weights_log2']['output']
    
    model.compile(
        optimizer = K.optimizers.Adam(learning_rate=lr_schedule),
        loss = loss,
        metrics = metrics,
        loss_weights = {k: 2 ** w[k] for k in ['pressure', 'diagnosis']}
    )

    return model