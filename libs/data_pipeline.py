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
    input_is_bad  = sig_is_bad[:sig_counts[0]]
    output_is_bad = sig_is_bad[sig_counts[0]:]
    output_is_bad |= tf.math.reduce_any(sigs[:, sig_counts[0]:] < 0, axis=0)
    
    if part == 'train':
        example_is_bad = tf.reduce_all(input_is_bad)
    else:
        val_input_is_bad = tf.boolean_mask(input_is_bad, validation_mask)
        example_is_bad = tf.reduce_any(val_input_is_bad)
    
    
    j_abp, j_pap = [H['output_sigs'].index(s) for s in ['ABP', 'PAP']]
    abp = sigs[:, sig_counts[0] + j_abp]
    pap = sigs[:, sig_counts[0] + j_pap]
    
    abp_too_small = tf.reduce_any(abp < 20)
    abp_diff_too_small = tf.reduce_max(abp) - tf.reduce_min(abp) < 5
    abp_sig_is_bad = output_is_bad[j_abp]
    abp_is_bad = abp_sig_is_bad | abp_too_small | abp_diff_too_small
    
    example_is_bad |= abp_is_bad
    example_is_good = ~example_is_bad
    
    if example_is_good and tf.reduce_any(sig_is_bad):
        sigs *= tf.cast(~sig_is_bad, sigs.dtype)
        sigs = tf.where(tf.math.is_nan(sigs), 0.0, sigs)
    
    hypertensive = [
        tf.reduce_max(abp) > 140 or tf.reduce_min(abp) > 90,
        tf.reduce_mean(pap) > 30
    ]
    hypertensive = [1 if i else -1 for i in hypertensive]
    hypertensive[0] = 0 if abp_is_bad else hypertensive[0]
    hypertensive[1] = 0 if output_is_bad[j_abp] else hypertensive[1]
            
    x, y = tf.split(sigs, sig_counts, axis=1)
    y = y[-2**H['pressure_window_log2']:]
    y = tf.stack([tf.reduce_max(y, axis=0), tf.reduce_min(y, axis=0)])
    
    if example_is_good and part != 'train':
        x = tf.where(validation_mask, x, 0.0)
    
    mask = tf.where(sig_is_bad, False, example.pop('mask'))
    mask = mask[:sig_counts[0]]
    
    x = {'signals': x, 'mask': mask}
    y = {**example, 'pressure': y, 'is_good': example_is_good}
    
    y['diagnosis'] = tf.concat([y['diagnosis'], hypertensive], axis=0)
        
    return x, y


def get_window_index_matrix(H):
    I_heart = tf.range(-2**H['window_size_log2'], 0) + 1
    I_resp = I_heart * initialize.RESP_SCALE
    S = H['input_sigs'] + H['output_sigs']
    I_samp = tf.gather([I_heart, I_resp], [int(s == 'RESP') for s in S])
    I_samp = tf.transpose(I_samp)
    I_sig = tf.range(len(S))
    I_sig = tf.broadcast_to(I_sig, [2**H['window_size_log2']] + I_sig.shape)
    I = tf.stack([I_samp, I_sig], axis=-1)
    I = tf.broadcast_to(I, [2**H['windows_per_chunk_log2']] + I.shape)
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
        shape = [2**H['windows_per_chunk_log2']],
        minval = 2**H['window_size_log2'] * initialize.RESP_SCALE, 
        maxval = prepare_data.CHUNK_SIZE, 
        dtype = 'int32'
    )
    window_ids = tf.cast(window_ids, 'uint16')

    example['rec_id'] = rec_id = rec_ids[i]
    example['seg_id'] = seg_id = seg_ids[i][j]
    adc_gain  = example.pop('adc_gain')[i][j]
    baseline  = example.pop('baseline')[i][j]
    sig_index = example.pop('sig_index')[i][j]
    example['mask'] = sig_index > 0
    
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
        key: tf.stack([example[key]] * 2**H['windows_per_chunk_log2']) 
        for key in example
    }
    examples = {'sigs': windows, **metadata}
    examples = tf.data.Dataset.from_tensor_slices(examples)
    return examples


def filter_example(x, y):
    return y.pop('is_good')


def build(H, tensors, part):
    n = next(iter(tensors.values())).shape[0]
    key = 'batch_size' if part == 'train' else 'batch_size_validation'
    batch_size = 2 ** H[key + '_log2']
    buffer_size = 2 ** H['batch_buffer_size_log2'] if part == 'train' else 1
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    if part == 'validation' and H['batch_size_validation_log2'] > 9:
        dataset = dataset.repeat(2 ** (H['batch_size_validation_log2'] - 9))
    dataset = dataset.shuffle(n)
    window_index_matrix = get_window_index_matrix(H)
    dataset = dataset.interleave(
        partial(get_examples, H, window_index_matrix), 
        block_length = 1, 
        cycle_length = buffer_size * batch_size,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )
    
    S_train, S_val = H['input_sigs'], H['input_sigs_validation']
    validation_mask = tf.constant([i in S_val for i in S_train], dtype='bool')
    
    dataset = dataset.map(partial(process_example, H, part, validation_mask))
    dataset = dataset.filter(filter_example)
    buffer_size *= batch_size * 2 ** H['windows_per_chunk_log2']
    dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size).repeat()
    return dataset