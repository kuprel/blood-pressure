import tensorflow as tf
from functools import partial, reduce

import initialize
import prepare_data

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)

def filter_datum(x, y, is_good):
    return is_good

def drop_filter_label(x, y, is_good):
    return x, y

def process_raw(H, is_validation, validation_mask, sigs):
    
    sig_has_nan = tf.math.reduce_any(tf.math.is_nan(sigs), axis=0)
    sig_has_nan |= tf.math.reduce_any(sigs < -100, axis=0)
    
    sig_diff_is_small = tf.reduce_max(sigs, axis=0) - tf.reduce_min(sigs, axis=0) < 0.01

    saturated = tf.abs(sigs - tf.reduce_max(sigs, axis=0)) < 1e-4
    saturated |= tf.abs(sigs - tf.reduce_min(sigs, axis=0)) < 1e-4
    saturated_count = tf.reduce_sum(tf.cast(saturated, 'int32'), axis=0)
    sig_is_saturated = saturated_count > sigs.shape[0] // 6
    
    bp = sigs[:, -1]
    bp_too_small = tf.reduce_any(bp < 20)
    bp_diff_is_small = tf.reduce_max(bp) - tf.reduce_min(bp) < 5
    
    sig_is_bad = sig_has_nan | sig_diff_is_small | sig_is_saturated
    
    if is_validation:
        example_is_bad = tf.reduce_any(tf.boolean_mask(sig_is_bad[:-1], validation_mask))
    else:
        example_is_bad = tf.reduce_all(sig_is_bad[:-1])
    
    example_is_bad |= sig_is_bad[-1] | bp_too_small | bp_diff_is_small
    
    example_is_good = ~example_is_bad
    
    if example_is_good and tf.reduce_any(sig_is_bad):
        sigs *= tf.cast(~sig_is_bad, sigs.dtype)
        sigs = tf.where(tf.math.is_nan(sigs), tf.constant(0, dtype=sigs.dtype), sigs)
    
    x, y = sigs[:,:-1], sigs[:,-1]
    y = y[-H['pressure_window']:]
    y = tf.stack([tf.reduce_max(y), tf.reduce_min(y)])
    
    if example_is_good and is_validation:
        x = tf.where(validation_mask, x, tf.constant(0, dtype=x.dtype))
    
    return x, y, example_is_good

def get_window_index_matrix(H):
    I_heart = tf.range(-H['window_size'], 0) + 1
    I_resp = I_heart * initialize.RESP_SCALE
    S = H['input_sigs_train'] + H['output_sigs']
    I_samp = tf.gather([I_heart, I_resp], [int(s == 'RESP') for s in S])
    I_samp = tf.transpose(I_samp)
    I_sig = tf.range(len(S))
    I_sig = tf.broadcast_to(I_sig, [H['window_size']] + I_sig.shape)
    I = tf.stack([I_samp, I_sig], axis=-1)
    I = tf.broadcast_to(I, [H['windows_per_chunk']] + I.shape)
    return I

def get_offsets(indices):
    offsets = tf.stack([indices, tf.zeros(shape=indices.shape[-1:], dtype='int32')], axis=1)
    offsets = tf.expand_dims(tf.expand_dims(offsets, 1), 1)
    return offsets

def get_windows(H, window_index_matrix, chunk_path, sig_indices, window_indices, baseline, gain):
    sigs = tf.io.read_file(chunk_path)
    sigs = tf.io.parse_tensor(sigs, out_type='int16')
    sigs.set_shape((prepare_data.CHUNK_SIZE, None))
    sigs = tf.concat([tf.zeros(shape=[prepare_data.CHUNK_SIZE, 1], dtype='int16'), sigs], axis=1)
    sigs = tf.gather(sigs, tf.cast(sig_indices, 'int32'), axis=1)
    sig_shape = (prepare_data.CHUNK_SIZE, len(H['input_sigs_train'] + H['output_sigs']))
    sigs.set_shape(sig_shape)
    sigs = tf.cast(tf.cast(sigs, 'int32') - baseline, 'float32') / gain
    offsets = get_offsets(window_indices)
    windows = tf.gather_nd(sigs, window_index_matrix + offsets)
    data = tf.data.Dataset.from_tensor_slices(windows)
    return data

def build(H, tensors, is_validation=False):
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    window_index_matrix = get_window_index_matrix(H)
    dataset = dataset.interleave(
        partial(get_windows, H, window_index_matrix), 
        block_length = 1, 
        cycle_length = H['batch_buffer_size'] * H['batch_size'],
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )
    validation_mask = [i in H['input_sigs_validation'] for i in H['input_sigs_train']]
    validation_mask = tf.constant(validation_mask, dtype=tf.bool, shape=[len(H['input_sigs_train'])])
    dataset = dataset.map(partial(process_raw, H, is_validation, validation_mask))
    if H['filter_data']:
        dataset = dataset.filter(filter_datum).map(drop_filter_label)
    buffer_size = H['batch_buffer_size'] * H['batch_size'] * H['windows_per_chunk']
    dataset = dataset.shuffle(buffer_size).batch(H['batch_size'])
    return dataset
