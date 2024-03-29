import tensorflow as tf
from tensorflow import keras as K
from functools import partial
import icd_util
from sklearn import metrics as skmetrics


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
    y_true, y_pred = y_true[:, j], y_pred[:, j]
    s1 = sensitivity(y_true, y_pred)
    s2 = specificity(y_true, y_pred)
    return (s1 + s2) / 2

 
def diagnosis_loss(j, y_true, y_pred):
    y_true, y_pred = y_true[:, j:j+1], y_pred[:, j:j+1]
#     y_true = tf.expand_dims(y_true, axis=1)
#     y_pred = tf.expand_dims(y_pred, axis=1)
    return custom_loss(y_true, y_pred)

def positive_loss(y_true, y_pred):
    mask = tf.cast(y_true == 1, 'float32')
    cross_entropy = -tf.math.log(y_pred) * mask
    counts = tf.maximum(tf.reduce_sum(mask, axis=0), 1)
    
#     loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=0) / counts)
    loss = tf.reduce_sum(cross_entropy, axis=0) / counts
    return loss


def negative_loss(y_true, y_pred):
    mask = tf.cast(y_true == -1, 'float32')
    cross_entropy = -tf.math.log(1 - y_pred) * mask
    counts = tf.maximum(tf.reduce_sum(mask, axis=0), 1)
    
#     loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=0) / counts)
    loss = tf.reduce_sum(cross_entropy, axis=0) / counts
    return loss


def custom_loss_old(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    loss_positive = positive_loss(y_true, y_pred)
    loss_negative = negative_loss(y_true, y_pred)
    
#     loss = (loss_positive + loss_negative) / 2
    loss = tf.reduce_mean(loss_positive + loss_negative)
    return loss   

def custom_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    log_y_pred = tf.math.log(y_pred)
    cross_entropy_pos = -tf.math.log(y_pred) * y_true
    cross_entropy_neg = -tf.math.log(1 - y_pred) * (1 - y_true)
    loss = tf.reduce_sum(loss_positive + loss_negative)
    return loss   

    
def build(H, diagnosis_codes):
    
    group_names = icd_util.load_group_strings()
    
    def get_name(code):
        if code not in group_names:
            return code
        name = code + '_' + group_names[code]
        name = name.replace(' ', '_')
        name = name.replace('/', '_')
        for character in "',()[]":
            name = name.replace(character, '')
        return name
    
    metric_base_funcs = {
        'accuracy': diagnosis_accuracy,
        'loss': diagnosis_loss,
#         'auc': roc_auc_score
    }
    
    metrics = {}
    
#     for k in metric_base_funcs:
#         metric_base_func = metric_base_funcs[k]
#         for j, code in enumerate(diagnosis_codes):
# #             def _metric(j, y_true, y_pred):
# #                 return metric_base_func(y_true[:, j], y_pred[:, j])
#             metrics[get_name(code) + '_' + k] = lambda y, y_: metric_base_func(y[:, j], y_[:, j])
    
    metrics = {
        get_name(code) + '_' + k: partial(metric_base_funcs[k], j) 
        for j, code in enumerate(diagnosis_codes) for k in metric_base_funcs 
    }
    
    for k in metrics:
        metrics[k].__name__ = k
    
    metrics = {'diagnosis': list(metrics.values())}
#     loss = {'diagnosis': custom_loss}
    
    return metrics