import tensorflow as tf
from tensorflow import keras as K
from functools import partial


CODE_NAMES = {
    '4019':  'hypertensive',
    '4280':  'congestive_heart_failure',
    '42731': 'atrial_fibrillation',
    '41401': 'coronary_atherosclerosis',
    '2720':  'hypercholesterolemia',
    '2724':  'hyperlipidemia',
    '5859':  'chronic_kidney_disease',
    '25000': 'diabetes',
    '5849':  'acute_kidney_failure',
    '51881': 'acute_resp_failure',
    '2859':  'anemia',
    '4552':  'post_hemorrhage_anemia',
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
    '431':   'intracerebral_hemorrhage',
    '4917':  'hypothyroidism',
    '2761':  'hyposomality',
    'V5867': 'insulin_user',
    '42789': 'other_cardiac_dysrhythmia',
    '4160':  'pulmonary_hypertension'
}


def accuracy(y_true, y_pred, threshold=0.5):
    y_true, y_pred = tf.cast(y_true, 'float32'), tf.cast(y_pred, 'float32')
    p = K.metrics.binary_accuracy(y_true, y_pred, threshold=threshold)
    p = tf.cond(tf.math.is_nan(p), lambda: 0.0, lambda: p)
    return p


def sensitivity(y_true, y_pred, threshold=0.5):
    mask = lambda y: tf.boolean_mask(y, y_true == 1)
    y_true, y_pred = mask(y_true), mask(y_pred)
    return accuracy(y_true, y_pred, threshold)


def specificity(y_true, y_pred):
    mask = lambda y: tf.boolean_mask(y, y_true == -1)
    y_true, y_pred = mask(y_true), mask(y_pred)
    y_true += 1
    return accuracy(y_true, y_pred)


def diagnosis_accuracy(j, y_true, y_pred):
    s1 = sensitivity(y_true[:, j], y_pred[:, j])
    s2 = specificity(y_true[:, j], y_pred[:, j])
    return (s1 + s2) / 2


def diagnosis_sensitivity(j, y_true, y_pred):
    return sensitivity(y_true[:, j], y_pred[:, j])


def diagnosis_specificity(j, y_true, y_pred):
    return specificity(y_true[:, j], y_pred[:, j])


def precise_threshold(j, y_true, y_pred, precision=0.8):
    mask = lambda y: tf.boolean_mask(y[:, j], y_true[:, j] != 0)
    y_true, y_pred = mask(y_true), mask(y_pred)
    y_true = y_true == 1
    I = tf.argsort(y_pred, direction='DESCENDING')
    y_true, y_pred = tf.gather(y_true, I), tf.gather(y_pred, I)
    precisions = tf.math.cumsum(tf.cast(y_true, 'float32'))
    precisions /= tf.cast(tf.range(tf.shape(y_true)[0]) + 1, 'float32')
    i_thresh = tf.reduce_sum(tf.cast(precisions > precision, 'int32'))
    i_thresh = tf.minimum(i_thresh, tf.shape(y_true)[0] - 1)
    threshold = tf.cond(i_thresh <= 0, lambda: 1., lambda: y_pred[i_thresh])
    return threshold


def precise_sensitivity(j, y_true, y_pred, precision=0.8):
    threshold = precise_threshold(j, y_true, y_pred, precision)
    return sensitivity(y_true[:, j], y_pred[:, j], threshold)


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


# def diagnosis_loss(H, y_true, y_pred, sample_weight=None):
#     y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
#     mask_pos = tf.cast(y_true ==  1, 'float32')
#     mask_neg = tf.cast(y_true == -1, 'float32')
#     xent_pos = -tf.math.log(y_pred) * mask_pos
#     xent_neg = -tf.math.log(1 - y_pred) * mask_neg
#     cross_entropy = xent_pos + xent_neg / H['positive_example_weight']
#     counts = tf.maximum(tf.reduce_sum(mask_pos + mask_neg, axis=0), 1)
#     loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=0) / counts)
#     return loss


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


def get_pressure_weight_matrix(H):
    w = H['loss_weights_log2']
    w1 = [2 ** w['metric'][k] for k in ['systolic', 'diastolic']]
    w2 = [2 ** w['sig'][s] for s in H['output_sigs']]
    w1 = tf.constant(w1, dtype='float32')
    w2 = tf.constant(w2, dtype='float32')
    W = tf.expand_dims(w1, axis=1) * tf.expand_dims(w2, axis=0)
    return W


def pressure_loss(H, W, y_true, y_pred, sample_weight=None):
    eps = 2**H['relative_target_radius_log2']
    L1 = tf.abs(y_pred - (1 - eps) * y_true)
    L2 = tf.abs(y_pred - (1 + eps) * y_true)
    mask = tf.cast(y_true != 0, 'float32')
    L = tf.reduce_sum([W] * mask * (L1 + L2), axis=0)
    counts = tf.maximum(tf.reduce_sum(mask, axis=0), 1)
    loss = tf.reduce_mean(L / counts) / 2
    return loss

    
def build(H, diagonsis_codes):
        
    pressure_metrics = {
        'systolic': systolic_error,
        'diastolic': diastolic_error,
#         'pulse': pulse_error,
    }
    
    diagnosis_metrics = {
        'sensitivity': diagnosis_sensitivity,
        'specificity': diagnosis_specificity,
        'accuracy': diagnosis_accuracy,
#         'precise_sensitivity': precise_sensitivity,
#         'precise_threshold': precise_threshold
    }
    
    get_code_name = lambda c: c + '_' + CODE_NAMES[c] if c in CODE_NAMES else c
    
    diagnosis_metrics = {
        get_code_name(code) + '_' + k: partial(diagnosis_metrics[k], j) 
        for j, code in enumerate(diagonsis_codes) for k in diagnosis_metrics 
    }
    
    pressure_metrics = {
        s + '_' + k: partial(pressure_metrics[k], j) 
        for j, s in enumerate(H['output_sigs']) for k in pressure_metrics 
    }
    
    metrics = {'pressure': pressure_metrics, 'diagnosis': diagnosis_metrics}
    
    for i in metrics:
        for j in metrics[i]:
            metrics[i][j].__name__ = j
        metrics[i] = list(metrics[i].values())
    
    W = get_pressure_weight_matrix(H)
    
    loss = {
        'pressure': partial(pressure_loss, H, W),
        'diagnosis': diagnosis_loss,
#         'diagnosis': partial(diagnosis_loss, H),
    }
    
    return loss, metrics