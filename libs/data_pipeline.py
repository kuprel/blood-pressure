import tensorflow as tf
from functools import partial, reduce
import numpy
import initialize
import prepare_data

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)


def process_example(H, part, validation_mask, example):
    
    sigs = example.pop('sigs')
    
    sig_has_nan = tf.math.reduce_any(tf.math.is_nan(sigs), axis=0)
    sig_has_nan |= tf.math.reduce_any(sigs < -100, axis=0)
    
    sig_max = tf.reduce_max(sigs, axis=0)
    sig_min = tf.reduce_min(sigs, axis=0)
    sig_diff_is_small = sig_max - sig_min < 0.01

    saturated = tf.abs(sigs - tf.reduce_max(sigs, axis=0)) < 1e-4
    saturated |= tf.abs(sigs - tf.reduce_min(sigs, axis=0)) < 1e-4
    saturated_count = tf.reduce_sum(tf.cast(saturated, 'int32'), axis=0)
    sig_is_saturated = saturated_count > sigs.shape[0] // 6
    
    sig_counts = [len(H['input_sigs']), len(H['output_sigs'])]
    
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
    
    x = {'signals': x, 'mask': example.pop('mask')}
    y = {**example, 'pressure': y, 'is_good': example_is_good}
        
    return x, y


def get_window_index_matrix(H):
    I_heart = tf.range(-H['window_size'], 0) + 1
    I_resp = I_heart * initialize.RESP_SCALE
    S = H['input_sigs'] + H['output_sigs']
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


def zfill(i, n):
    return tf.strings.substr('0'*n + tf.as_string(i), -n, n)


def get_examples(H, W, example):
    
    rand_int = lambda maxval: tf.random.uniform([1], 0, maxval, 'int32')[0]
    length = lambda tensor: tf.shape(tensor)[0]
    
    rec_ids = example.pop('rec_id')[0]
    seg_ids = example.pop('seg_id')
    
    i = rand_int(length(rec_ids))
    j = rand_int(length(seg_ids[i]))
    
    chunk_count = tf.cast(example.pop('chunk_count')[i][j], 'int32')
    chunk_id = rand_int(chunk_count)
    
    window_ids = tf.random.uniform(
        shape = [H['windows_per_chunk']],
        minval = H['window_size'] * initialize.RESP_SCALE, 
        maxval = prepare_data.CHUNK_SIZE, 
        dtype = 'int32'
    )
    window_ids = tf.cast(window_ids, 'uint16')

    rec_id = rec_ids[i]
    seg_id = seg_ids[i][j]
    adc_gain  = example.pop('adc_gain')[i][j]
    baseline  = example.pop('baseline')[i][j]
    sig_index = example.pop('sig_index')[i][j]
    example['mask'] = sig_index[:len(H['input_sigs'])] > 0
    
    for key in ['diagnosis', 'age', 'weight', 'height']:
        example[key] = example[key][0][i]
        
    chunk_name = tf.as_string(rec_id) 
    chunk_name += '_' + zfill(seg_id, 4)
    chunk_name += '_' + zfill(chunk_id, 4)
        
    sigs_path = prepare_data.ROOT_SERIAL + chunk_name + '.tfrec'
    sigs = tf.io.read_file(sigs_path)
    sigs = tf.io.parse_tensor(sigs, out_type='int16')
    sigs.set_shape((prepare_data.CHUNK_SIZE, None))
    zeros = tf.zeros(shape=[prepare_data.CHUNK_SIZE, 1], dtype='int16')
    sigs = tf.concat([zeros, sigs], axis=1)
    sigs = tf.gather(sigs, tf.cast(sig_index, 'int32'), axis=1)
    sig_count = len(H['input_sigs'] + H['output_sigs'])
    sigs.set_shape((prepare_data.CHUNK_SIZE, sig_count))
    sigs = tf.cast(sigs - baseline, 'float32') / adc_gain
        
    dW = tf.cast(get_offsets(window_ids), 'int32')
    windows = tf.gather_nd(sigs, W + dW)
    metadata = {
        key: tf.stack([example[key]] * H['windows_per_chunk']) 
        for key in example
    }
    examples = {'sigs': windows, **metadata}
    examples = tf.data.Dataset.from_tensor_slices(examples)
    return examples


def filter_example(x, y):
    return y.pop('is_good')


def build(H, tensors, part):
    n = next(iter(tensors.values())).shape[0]
    m = H['batch_size'] if part == 'train' else 2**16
    buffer_size = H['batch_buffer_size'] if part == 'train' else 1
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.shuffle(n)
    window_index_matrix = get_window_index_matrix(H)
    dataset = dataset.interleave(
        partial(get_examples, H, window_index_matrix), 
        block_length = 1, 
        cycle_length = buffer_size * m,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )
    
    S_train, S_val = H['input_sigs'], H['input_sigs_validation']
    validation_mask = tf.constant([i in S_val for i in S_train], dtype='bool')
    
    dataset = dataset.map(partial(process_example, H, part, validation_mask))
    dataset = dataset.filter(filter_example)
    buffer_size *= m * H['windows_per_chunk']
    dataset = dataset.shuffle(buffer_size).batch(m).repeat()
    return dataset