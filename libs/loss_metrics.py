import tensorflow as tf
from tensorflow import keras as K
from functools import partial


MIN_PPV = 0.9


def get_pressure_weight_matrix(H):
    w1 = [H['loss_weights']['metric'][k] for k in ['systolic', 'diastolic']]
    w2 = [H['loss_weights']['sig'][s] for s in H['output_sigs']]
    w1 = tf.constant(w1, dtype='float32')
    w2 = tf.constant(w2, dtype='float32')
    W = tf.expand_dims(w1, axis=1) * tf.expand_dims(w2, axis=0)
    return W


def pressure_loss(H, y_true, y_pred, sample_weight=None):
    eps = H['relative_target_radius']
    L1 = tf.abs(y_pred - (1 - eps) * y_true)
    L2 = tf.abs(y_pred - (1 + eps) * y_true)
    mask = tf.cast(y_true != 0, 'float32')
    W = get_pressure_weight_matrix(H)
    L = tf.reduce_sum([W] * mask * (L1 + L2), axis=0)
    counts = tf.maximum(tf.reduce_sum(mask, axis=0), 1)
    loss = tf.reduce_mean(L / counts) / 2
    return loss


def positive_diagnosis_loss(y_true, y_pred):
    mask = tf.cast(y_true == 1, 'float32')
    cross_entropy = -tf.math.log(y_pred) * mask
    counts = tf.maximum(tf.reduce_sum(mask, axis=0), 1)
    loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=0) / counts)
    return loss


def negative_diagnosis_loss(y_true, y_pred):
    mask = tf.cast(y_true == -1, 'float32')
    cross_entropy = -tf.math.log(1 - y_pred) * mask
    counts = tf.maximum(tf.reduce_sum(mask, axis=0), 1)
    loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=0) / counts)
    return loss


def diagnosis_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    loss_positive = positive_diagnosis_loss(y_true, y_pred)
    loss_negative = negative_diagnosis_loss(y_true, y_pred)
    return (loss_positive + loss_negative) / 2


def to_hypertensive(y):
    y = tf.reduce_any(y[:, :, 0] > [140., 90.], axis=1)
    y = tf.cast(y, 'int32')
    y = tf.where(y == 0, -1, y)
    return y


def hypertensive_sensitivity(y_true, y_pred):
    return sensitivity(to_hypertensive(y_true), to_hypertensive(y_pred))


def hypertensive_specificity(y_true, y_pred):
    return specificity(to_hypertensive(y_true), to_hypertensive(y_pred))


def hypertensive_accuracy(y_true, y_pred):
    s1 = hypertensive_sensitivity(y_true, y_pred)
    s2 = hypertensive_specificity(y_true, y_pred)
    return (s1 + s2) / 2


def systolic_error(j, y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y[:, 0, j], y_true[:, 0, j] != 0)
    error = K.losses.mean_absolute_error(mask(y_true), mask(y_pred))
    error = tf.cond(tf.math.is_nan(error), lambda: 0.0, lambda: error)
    return error


def diastolic_error(j, y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y[:, 1, j], y_true[:, 0, j] != 0)
    error = K.losses.mean_absolute_error(mask(y_true), mask(y_pred))
    error = tf.cond(tf.math.is_nan(error), lambda: 0.0, lambda: error)
    return error


def pulse_error(j, y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y[:,0,j] - y[:,1,j], y_true[:,0,j] != 0)
    error = K.losses.mean_absolute_error(mask(y_true), mask(y_pred))
    error = tf.cond(tf.math.is_nan(error), lambda: 0.0, lambda: error)
    return error


def diagnosis_sensitivity(j, y_true, y_pred):
    return sensitivity(y_true[:, j], y_pred[:, j])


def diagnosis_specificity(j, y_true, y_pred):
    return specificity(y_true[:, j], y_pred[:, j])


def _precise_threshold(y_true, y_pred):
    I = tf.argsort(y_pred, direction='DESCENDING')
    y_true, y_pred = tf.gather(y_true, I), tf.gather(y_pred, I)
    ppv = tf.math.cumsum(tf.cast(y_true, 'float32'))
    ppv /= tf.cast(tf.range(tf.shape(y_true)[0]) + 1, 'float32')
    i_thresh = tf.reduce_sum(tf.cast(ppv > MIN_PPV, 'int32'))
    i_thresh = tf.minimum(i_thresh, tf.shape(y_true)[0] - 1)
    thresh = tf.cond(i_thresh == 0, lambda: 1., lambda: y_pred[i_thresh])
    return thresh


def _precise_sensitivity(y_true, y_pred):
    threshold = _precise_threshold(y_true, y_pred)
    mask = lambda y: tf.boolean_mask(y, y_true)
    y_true, y_pred = mask(y_true), mask(y_pred)
    return accuracy(y_true, y_pred, threshold)


def precise_threshold(j, y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y[:, j], y_true[:, j] != 0)
    y_true, y_pred = mask(y_true), mask(y_pred)
    y_true = y_true == 1
    return _precise_threshold(y_true, y_pred)


def precise_sensitivity(j, y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y[:, j], y_true[:, j] != 0)
    y_true, y_pred = mask(y_true), mask(y_pred)
    y_true = y_true == 1
    return _precise_sensitivity(y_true, y_pred)


def diagnosis_accuracy(j, y_true, y_pred):
    s1 = sensitivity(y_true[:, j], y_pred[:, j])
    s2 = specificity(y_true[:, j], y_pred[:, j])
    return (s1 + s2) / 2


def true_positive_loss(y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y, y_true == 1)
    y, y_ = mask(y_true), mask(y_pred)
    loss = K.metrics.binary_crossentropy(y, y_)
    loss = tf.cond(tf.math.is_nan(loss), lambda: 0.0, lambda: loss)
    return loss


def true_negative_loss(y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y, y_true == -1)
    y, y_ = mask(y_true), mask(y_pred)
    y += 1
    loss = K.metrics.binary_crossentropy(y, y_)
    loss = tf.cond(tf.math.is_nan(loss), lambda: 0.0, lambda: loss)
    return loss


def binary_outcome_loss(y_true, y_pred):
    loss_positive = true_positive_loss(y_true, y_pred)
    loss_negative = true_negative_loss(y_true, y_pred)
    loss = (loss_positive + loss_negative) / 2
    return loss


def accuracy(y_true, y_pred, threshold=0.5):
    y_true, y_pred = tf.cast(y_true, 'float32'), tf.cast(y_pred, 'float32')
    p = K.metrics.binary_accuracy(y_true, y_pred, threshold=threshold)
    p = tf.cond(tf.math.is_nan(p), lambda: 0.0, lambda: p)
    return p


def sensitivity(y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y, y_true == 1)
    y_true, y_pred = mask(y_true), mask(y_pred)
    return accuracy(y_true, y_pred)


def specificity(y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y, y_true == -1)
    y_true, y_pred = mask(y_true), mask(y_pred)
    y_true += 1
    return accuracy(y_true, y_pred)


def average_accuracy(y_true, y_pred, sample_weight=None):
    s1 = sensitivity(y_true, y_pred)
    s2 = specificity(y_true, y_pred)
    return (s1 + s2) / 2

    
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
    
    pressure_metrics = list(pressure_metrics.values())
    pressure_metrics += [
        hypertensive_sensitivity, 
        hypertensive_specificity,
        hypertensive_accuracy
    ]
    
    diagnosis_metrics = {
        'sensitivity': diagnosis_sensitivity,
        'specificity': diagnosis_specificity,
        'accuracy': diagnosis_accuracy,
        'precise_sensitivity': precise_sensitivity,
        'precise_threshold': precise_threshold
    }
    
    code_names = {
        '4019':  'hypertensive',
        '4280':  'congestive_heart_failure',
        '42731': 'atrial_fibrillation',
        '41401': 'coronary_atherosclerosis',
        '2724':  'hyperlipidemia',
        '5859':  'chronic_kidney_disease',
        '25000': 'diabetes',
        '5849':  'acute_kidney_failure',
        '51881': 'acute_resp_failure',
        '2859':  'anemia',
        '4240':  'mitral_valve_disorder',
        '4241':  'aortic_valve_disorder',
        '78552': 'septic_shock',
        '99592': 'severe_sepsis',
        '2762':  'acidosis',
        '5119':  'pleural_effusion',
        '41071': 'subendocardial_infarction',
        '27800': 'obesity',
        '27651': 'dehydration',
        '4275':  'cardiac_arrest',
        '5715':  'cirrhosis',
        '42732': 'atrial_flutter',
        '42832': 'chronic_diastolic_heart_failure',
        '42833': 'acute_diastolic_heart_failure',
        '79902': 'hypoxemia',
        '27652': 'hypovolemia',
        '431':   'intracerebral_hemorrhage'
    }
    
    diagnosis_metrics = {
        code_names[code] + '_' + k: partial(diagnosis_metrics[k], j) 
        for j, code in enumerate(H['icd_codes']) for k in diagnosis_metrics 
    }
    
    for k in diagnosis_metrics:
        diagnosis_metrics[k].__name__ = k
    
    metrics = {
        'pressure': pressure_metrics, 
        'gender': [sensitivity, specificity, average_accuracy],
        'died': [sensitivity, specificity, average_accuracy],
        'diagnosis': list(diagnosis_metrics.values()),
    }
    
    def gender_loss(y_true, y_pred):
        return binary_outcome_loss(y_true, y_pred)

    def died_loss(y_true, y_pred):
        return binary_outcome_loss(y_true, y_pred)
    
    loss = {
        'pressure': partial(pressure_loss, H),
        'gender': gender_loss,
        'died': died_loss,
        'diagnosis': diagnosis_loss,
    }
    
    return loss, metrics