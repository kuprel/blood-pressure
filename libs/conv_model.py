import tensorflow as tf
from tensorflow import keras as K
from functools import partial, reduce

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)

def systolic_error(bp_true, bp_pred):
    return K.losses.mean_absolute_error(bp_true[:, 0], bp_pred[:, 0])

def diastolic_error(bp_true, bp_pred):
    return K.losses.mean_absolute_error(bp_true[:, 1], bp_pred[:, 1])
    
def pulse_error(bp_true, bp_pred):
    pp_true = bp_true[:, 0] - bp_true[:, 1]
    pp_pred = bp_pred[:, 0] - bp_pred[:, 1]
    return K.losses.mean_absolute_error(pp_true, pp_pred)

def target_region_loss(r, y_true, y_pred):
    errors = y_pred - y_true
    losses = tf.abs(errors - r) + tf.abs(errors + r)
    loss = tf.reduce_mean(losses) / 2 - r
    return loss

def loss(H, y_true, y_pred, sample_weight):
    keys = ['systolic', 'diastolic', 'pulse']
    w_sys, w_dia, w_pp = [H['loss_weight'][i] for i in keys]
    r_sys, r_dia, r_pp = [H['target_radius'][i] for i in keys]
    pp_true = y_true[:, 0] - y_true[:, 1]
    pp_pred = y_pred[:, 0] - y_pred[:, 1]
    loss_sys = target_region_loss(r_sys,  y_true[:, 0], y_pred[:, 0])
    loss_dia = target_region_loss(r_dia,  y_true[:, 1], y_pred[:, 1])
    loss_pp  = target_region_loss(r_pp,  pp_true,      pp_pred)
    loss = loss_sys * w_sys + loss_dia * w_dia + loss_pp * w_pp
    return loss

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
    

def build(H):

    m = len(H['input_sigs_train'])
    x = K.layers.Input(shape=(H['window_size'], m), batch_size=H['batch_size'])
    
    mask = ~tf.reduce_all(x == 0, axis=1, keepdims=True)
    mask = tf.cast(mask, dtype=x.dtype)
    
    bp_layer = K.layers.Dense(2)
    
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
        bp_layer,
    ]
    
    net = compose(*layers[::-1])
    
    model = K.models.Model(inputs=x, outputs=net(x))
    
    bp_layer.set_weights([
        bp_layer.get_weights()[0],
        tf.constant(H['mean_bp'], dtype='float32')
    ])  

    lr_schedule = K.optimizers.schedules.PiecewiseConstantDecay(
        boundaries = [H['steps_per_epoch'] * i for i in H['lr_boundaries']],
        values = [H['learning_rate'] / i for i in H['lr_divisors']]
    )
        
    model.compile(
        optimizer = K.optimizers.Adam(learning_rate=lr_schedule),
        loss=partial(loss, H),
        metrics=[systolic_error, diastolic_error, pulse_error]
    )

    return model