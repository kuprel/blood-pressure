import tensorflow as tf
from tensorflow import keras as K
from functools import partial, reduce

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)

def resnet_block_a(H):
    conv_args = {
        'filters': H['filter_count'], 
        'activation': None,
        'padding': 'same',
        'use_bias': False,
        'kernel_size': (H['kernel_sizes'][1], 1), 
        'strides': (H['strides'][1], 1),   
    }
    
    layers = [
        K.layers.BatchNormalization(center=H['use_bias_a']),
        K.layers.Activation('relu'),
        K.layers.Conv2D(**conv_args),
        K.layers.BatchNormalization(center=H['use_bias_a']),
        K.layers.Activation('relu'),
        K.layers.Conv2D(**conv_args),
    ]
    
    f = compose(*layers[::-1])
    block = lambda x: f(x) + x[:, ::conv_args['strides'][0]**2]
    return block


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


def resnet_block_ab(H):
    
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
        K.layers.BatchNormalization(center=H['use_bias_a']),
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


def first_layer(H):
    args = {
        'filters': H['filter_count'],
        'padding': 'same',
        'activation': 'relu',
        'use_bias': H['use_bias_a'],
        'kernel_size': (H['kernel_sizes'][0], 1), 
        'strides': (H['strides'][0], 1),
    }
    layers = [
        partial(tf.expand_dims, axis=-1),
        K.layers.Conv2D(**args)
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
    

def build(H):

    x = K.layers.Input(shape=(H['window_size'], len(H['input_sigs_train'])))
    
    pred_dim = 1 if H['predict_pulse_pressure'] else 2
    final_layer = K.layers.Dense(pred_dim)
    
    layers = [
        first_layer(H),
        resnet_block_a(H),
        resnet_block_ab(H),
        *[resnet_block_b(H) for i in range(H['layer_count'])],
        dense_layer(H),
        final_layer,
    ]
    
    net = compose(*layers[::-1])
    
    model = K.models.Model(inputs=x, outputs=net(x))

    mean_pred = [H['mean_pp']] if H['predict_pulse_pressure'] else H['mean_bp']
    
    final_layer.set_weights([
        final_layer.get_weights()[0],
        tf.constant(mean_pred, dtype='float32')
    ])  

    lr_schedule = K.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = [H['steps_per_epoch'] * i for i in H['lr_boundaries']],
        values = [H['learning_rate'] / i for i in H['lr_divisors']]
    )
    
    def pulse_pressure_error(y_true, y_pred):
        pp_true = y_true[:, 0] - y_true[:, 1]
        pp_pred = y_pred[:, 0] - y_pred[:, 1]
        pp_diff = pp_true - pp_pred
        pp_error = tf.reduce_mean(tf.abs(pp_diff))
        return pp_error
    
    if H['predict_pulse_pressure']:
        metrics = []
    else:
        metrics = [pulse_pressure_error]

    model.compile(
        optimizer = K.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_absolute_error',
        metrics=metrics
    )

    return model