import tensorflow as tf
from functools import partial, reduce
import numpy
import initialize
import prepare_data

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)


def process_sigs(H, part, validation_mask, sigs):
    
    sig_has_nan = tf.math.reduce_any(tf.math.is_nan(sigs), axis=0)
    sig_has_nan |= tf.math.reduce_any(sigs < -100, axis=0)
    
    sig_max = tf.reduce_max(sigs, axis=0)
    sig_min = tf.reduce_min(sigs, axis=0)
    sig_diff_is_small = sig_max - sig_min < 0.01

    saturated = tf.abs(sigs - tf.reduce_max(sigs, axis=0)) < 1e-4
    saturated |= tf.abs(sigs - tf.reduce_min(sigs, axis=0)) < 1e-4
    saturated_count = tf.reduce_sum(tf.cast(saturated, 'int32'), axis=0)
    sig_is_saturated = saturated_count > sigs.shape[0] // 6
    
    sig_counts = [len(H['input_sigs_train']), len(H['output_sigs'])]
    
    sig_is_bad = sig_has_nan | sig_diff_is_small | sig_is_saturated
    input_is_bad = sig_is_bad[:sig_counts[0]]
    
    if part == 'train':
        example_is_bad = tf.reduce_all(input_is_bad)
    else:
        val_input_is_bad = tf.boolean_mask(input_is_bad, validation_mask)
        example_is_bad = tf.reduce_any(val_input_is_bad)
    
    bp = sigs[:, sig_counts[0]]
    bp_too_small = tf.reduce_any(bp < 20)
    bp_diff_is_small = tf.reduce_max(bp) - tf.reduce_min(bp) < 5
    bp_sig_is_bad = sig_is_bad[sig_counts[0]]
    
    example_is_bad |= bp_sig_is_bad | bp_too_small | bp_diff_is_small
    
    example_is_good = ~example_is_bad
    
    if example_is_good and tf.reduce_any(sig_is_bad):
        sigs *= tf.cast(~sig_is_bad, sigs.dtype)
        zero = tf.constant(0, dtype=sigs.dtype)
        sigs = tf.where(tf.math.is_nan(sigs), zero, sigs)
    
    x, y = tf.split(sigs, sig_counts, axis=1)
    y = y[-H['pressure_window']:]
    
    y = tf.stack([tf.reduce_max(y, axis=0), tf.reduce_min(y, axis=0)])
    
    if example_is_good and part != 'train':
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
    zeros = tf.zeros(shape=indices.shape[-1:], dtype='uint16')
    offsets = tf.stack([indices, zeros], axis=1)
    offsets = tf.expand_dims(tf.expand_dims(offsets, 1), 1)
    return offsets


def get_windows(H, W, chunk_name, sig_indices, window_indices, baseline, gain):
    sigs = tf.io.read_file(prepare_data.ROOT_SERIAL + chunk_name + '.tfrec')
    sigs = tf.io.parse_tensor(sigs, out_type='int16')
    sigs.set_shape((prepare_data.CHUNK_SIZE, None))
    zeros = tf.zeros(shape=[prepare_data.CHUNK_SIZE, 1], dtype='int16')
    sigs = tf.concat([zeros, sigs], axis=1)
    sigs = tf.gather(sigs, tf.cast(sig_indices, 'int32'), axis=1)
    n_sig = len(H['input_sigs_train'] + H['output_sigs'])
    sigs.set_shape((prepare_data.CHUNK_SIZE, n_sig))
    sigs = tf.cast(sigs - baseline, 'float32') / gain
    dW = tf.cast(get_offsets(window_indices), 'int32')
    windows = tf.gather_nd(sigs, W + dW)
    data = tf.data.Dataset.from_tensor_slices(windows)
    return data


def filter_datum(x, y, is_good):
    return is_good


def drop_filter_label(x, y, is_good):
    return x, y


def build(H, data, part):
    I = numpy.random.permutation(data[0].shape[0])
    data = tuple([tf.gather(d, I) for d in data[:5]])
    dataset = tf.data.Dataset.from_tensor_slices(data)
    window_index_matrix = get_window_index_matrix(H)
    dataset = dataset.interleave(
        partial(get_windows, H, window_index_matrix), 
        block_length = 1, 
        cycle_length = H['batch_buffer_size'] * H['batch_size'],
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )
    
    # put this into process_sigs?
    S_train, S_val = H['input_sigs_train'], H['input_sigs_validation']
    validation_mask = tf.constant([i in S_val for i in S_train], dtype='bool')
    
    dataset = dataset.map(partial(process_sigs, H, part, validation_mask))
    if H['filter_data']:
        dataset = dataset.filter(filter_datum).map(drop_filter_label)
    buffer_size = H['batch_buffer_size'] 
    buffer_size *= H['batch_size'] * H['windows_per_chunk']
    dataset = dataset.shuffle(buffer_size).batch(H['batch_size'])
    return dataset
