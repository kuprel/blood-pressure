import tensorflow as tf
from tensorflow import keras as K
from functools import partial, reduce
import numpy
# import loss_metrics

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), reversed(F))

def first_layer(H):
    args = {
        'filters': 2 ** H['filter_counts_log2'][0],
        'padding': 'same',
        'activation': 'relu',
        'kernel_size': (2 ** H['kernel_sizes_log2'][0], 1),
#         'dilation_rate': (2 ** H['dilation_log2'], 1),
        'strides': (2 ** H['strides_log2'][0], 1),
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
        'kernel_regularizer': K.regularizers.l2(2**H['kernel_reg_log2'])
    }
        
    layer = K.layers.Conv2D(**args)
    return layer


def resnet_block_a(H):
    
    conv_args = {
        'filters': 2 ** H['filter_counts_log2'][0], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_size': (2 ** H['kernel_sizes_log2'][1], 1), 
        'dilation_rate': (2 ** H['dilation_log2'], 1),
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
        'kernel_regularizer': K.regularizers.l2(2**H['kernel_reg_log2'])
    }
    
    f = compose(
        K.layers.BatchNormalization(),
        K.layers.Activation('relu'),
        K.layers.Conv2D(**conv_args),
        K.layers.BatchNormalization(),
        K.layers.Activation('relu'),
        K.layers.Conv2D(**conv_args),
    )
    
    block = lambda x: f(x) + x
    return block
    

def resnet_block_ab(H):

    conv_args = {
        'filters': 2 ** H['filter_counts_log2'][0], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
        'kernel_regularizer': K.regularizers.l2(2**H['kernel_reg_log2']),
#         'dilation_rate': (2 ** H['dilation_log2'], 1),
        'kernel_size': (2 ** H['kernel_sizes_log2'][2], len(H['input_sigs'])), 
        'strides': (2 ** H['strides_log2'][1], len(H['input_sigs'])),
    }

    f = compose(
        K.layers.BatchNormalization(),
        K.layers.Activation('relu'),
        K.layers.Conv2D(**conv_args),
    )
    
    sum_stride = lambda x: tf.reduce_sum(x[:, ::2**H['strides_log2'][1]], axis=2)
    block = lambda x: tf.squeeze(f(x), axis=2) + sum_stride(x)
    return block   
    
    
def resnet_block_b(H):
    conv_args = {
        'filters': 2 ** H['filter_counts_log2'][0], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_size': 2 ** H['kernel_sizes_log2'][3],
        'dilation_rate': 2 ** H['dilation_log2'],
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
        'kernel_regularizer': K.regularizers.l2(2**H['kernel_reg_log2'])
    }
    
    f = compose(
        K.layers.BatchNormalization(),
        K.layers.Activation('relu'),
        K.layers.Conv1D(**conv_args),
        K.layers.BatchNormalization(),
        K.layers.Activation('relu'),
        K.layers.Conv1D(**conv_args),
    )
    
    block = lambda x: f(x) + x
    return block

def last_conv_layer(H):
    args = {
        'filters': 2 ** H['filter_counts_log2'][1],
        'activation': 'relu',
        'padding': 'same',
        'kernel_size': 2 ** H['kernel_sizes_log2'][4],
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
        'kernel_regularizer': K.regularizers.l2(2**H['kernel_reg_log2']),
        'strides': 2 ** H['strides_log2'][2]
    }
        
    layer = K.layers.Conv1D(**args)
    return layer


def dense_layer(H, name, output_size):
    args = {
        'units': output_size,
        'name': name,
        'activation': 'sigmoid',
        'kernel_initializer': K.initializers.RandomNormal(
            mean=0.0, 
            stddev=2**H['initial_weights_std_log2']
        ),
        'kernel_regularizer': K.regularizers.l2(2**H['kernel_reg_log2'])
    }
    return K.layers.Dense(**args)
    

def build(H, priors):

    outputs = []
    
    input_shape = (2**H['window_size_log2'], len(H['input_sigs']))
    signals = K.layers.Input(input_shape, name='signals')
    mask = K.layers.Input(input_shape[1], name = 'mask')
    
    mask_float = tf.cast(mask, dtype=signals.dtype)
    mask_float = tf.expand_dims(tf.expand_dims(mask_float, axis=1), axis=-1)
    
    z = tf.expand_dims(signals, axis=-1)
    z = first_layer(H)(z) * mask_float
    
    sig_groups = [slice(0, 1), slice(1, 2), slice(2, 6), slice(6, None)]
    Z = [z[:, :, G] for G in sig_groups]
    
    for i in range(2**H['layer_counts_log2'][0]):
        for j, (G, z) in enumerate(zip(sig_groups, Z)):
            Z[j] = resnet_block_a(H)(z)

    z = tf.concat(Z, axis=2)
    z = resnet_block_ab(H)(z)

    for i in range(2**H['layer_counts_log2'][1]):
        z = resnet_block_b(H)(z)

    z = last_conv_layer(H)(z)        
    z = K.layers.Flatten()(z)
#     features_dropped = K.layers.Dropout(2**H['dropout_log2'])(features)
    
    diagnosis_layer = dense_layer(H, name='diagnosis', output_size=len(priors))
    diagnosis = diagnosis_layer(z)
    outputs.append(diagnosis)
    
    model = K.models.Model(
        inputs=[signals, mask], 
        outputs=outputs[::-1]
    )
    
    diagnosis_layer.set_weights([
        diagnosis_layer.get_weights()[0],
        tf.math.log(priors) - tf.math.log(1 - priors)
    ])

    boundaries = [2**H['steps_per_epoch_log2'] * i for i in H['lr_boundaries']]
    values = [2**(H['learning_rate_log2'] - i) for i in H['lr_divisors_log2']]
    lr_schedule = K.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = boundaries,
        values = values
    )
    
#     codes = list(priors.index)
#     loss, metrics = loss_metrics.build(H, codes)
    
    model.compile(
        optimizer = K.optimizers.Adam(learning_rate=lr_schedule),
        loss = K.losses.BinaryCrossentropy(reduction=K.losses.Reduction.SUM_OVER_BATCH_SIZE)
#         metrics = metrics
    )

    return model