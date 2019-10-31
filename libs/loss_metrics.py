import tensorflow as tf
from tensorflow import keras as K
from functools import partial


def pressure_loss(H, y_true, y_pred, sample_weight):
    eps = H['relative_target_radius']
    L1 = tf.abs(y_pred - (1 - eps) * y_true)
    L2 = tf.abs(y_pred - (1 + eps) * y_true)
    w1 = [H['loss_weights']['metric'][k] for k in ['systolic', 'diastolic']]
    w2 = [H['loss_weights']['sig'][s] for s in H['output_sigs']]
    w1 = tf.constant(w1, dtype='float32')
    w2 = tf.constant(w2, dtype='float32')
    W = tf.expand_dims(w1, axis=1) * tf.expand_dims(w2, axis=0)
    W = tf.stack([W] * H['batch_size'])
    W = tf.where(y_true == 0, 0.0, W)
    L = tf.reduce_sum(tf.reduce_mean(W * (L1 + L2), axis=0)) / 2
    return L


def extract_present(y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y, y_true[:, 0] != 0, axis=0)
    return mask(y_true), mask(y_pred)


def extract_outcome_present(y_true, y_pred, outcome):
    mask = lambda y: tf.boolean_mask(y, y_true == outcome)
    return mask(y_true), mask(y_pred)


def systolic_error(j, y_true, y_pred):
    y, y_ = extract_present(y_true[:, :, j], y_pred[:, :, j])
    error = K.losses.mean_absolute_error(y[:, 0], y_[:, 0])
    error = tf.cond(tf.math.is_nan(error), lambda: 0.0, lambda: error)
    return error


def diastolic_error(j, y_true, y_pred):
    y, y_ = extract_present(y_true[:, :, j], y_pred[:, :, j])
    error = K.losses.mean_absolute_error(y[:, 1], y_[:, 1])
    is_nan = tf.math.is_nan(error)
    error = tf.cond(is_nan, lambda: tf.constant(0.0), lambda: error)
    return error


def pulse_error(j, y_true, y_pred):
    y, y_ = extract_present(y_true[:, :, j], y_pred[:, :, j])
    error = K.losses.mean_absolute_error(y[:, 0] - y[:, 1], y_[:, 0] - y_[:, 1])
    is_nan = tf.math.is_nan(error)
    error = tf.cond(is_nan, lambda: tf.constant(0.0), lambda: error)
    return error


def true_positive_loss(y_true, y_pred):
    y, y_ = extract_outcome_present(y_true, y_pred, 1)
    loss = K.metrics.binary_crossentropy(y, y_)
    loss = tf.cond(tf.math.is_nan(loss), lambda: 0.0, lambda: loss)
    return loss


def true_negative_loss(y_true, y_pred):
    y, y_ = extract_outcome_present(y_true, y_pred, -1)
    y += 1
    loss = K.metrics.binary_crossentropy(y, y_)
    loss = tf.cond(tf.math.is_nan(loss), lambda: 0.0, lambda: loss)
    return loss


def binary_outcome_loss(y_true, y_pred):
    loss_positive = true_positive_loss(y_true, y_pred)
    loss_negative = true_negative_loss(y_true, y_pred)
    loss = (loss_positive + loss_negative) / 2
    return loss


def accuracy(y_true, y_pred):
    p = K.metrics.binary_accuracy(tf.cast(y_true, 'float32'), y_pred)
    p = tf.cond(tf.math.is_nan(p), lambda: 0.0, lambda: p)
    return p


def sensitivity(y_true, y_pred):
    y_true, y_pred = extract_outcome_present(y_true, y_pred, 1)
    return accuracy(y_true, y_pred)


def specificity(y_true, y_pred):
    y_true, y_pred = extract_outcome_present(y_true, y_pred, -1)
    y_true += 1
    return accuracy(y_true, y_pred)


def build(H):
    pressure_metrics = {
        'systolic': systolic_error,
        'diastolic': diastolic_error,
        'pulse': pulse_error,
    }
    
    pressure_metrics = {
        s + '_' + k: partial(pressure_metrics[k], j) 
        for j, s in enumerate(H['output_sigs']) for k in pressure_metrics 
    }
    
    for k in pressure_metrics:
        pressure_metrics[k].__name__ = k
            
    metrics = {
        'pressure': list(pressure_metrics.values()), 
        'gender': [sensitivity, specificity],
        'died': [sensitivity, specificity]
    }
    
    def gender_loss(y_true, y_pred):
        return binary_outcome_loss(y_true, y_pred)

    def died_loss(y_true, y_pred):
        return binary_outcome_loss(y_true, y_pred)
    
    loss = {
        'pressure': partial(pressure_loss, H),
        'gender': gender_loss,
        'died': died_loss
    }
    
    return loss, metrics