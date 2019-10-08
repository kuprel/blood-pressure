import tensorflow as tf
from functools import partial, reduce

from Brett import data_util, prepare_data

compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)

S = prepare_data.INPUT_SIGS + prepare_data.OUTPUT_SIGS

def filter_datum(x, y):
    any_nan = compose(tf.math.reduce_any, tf.math.is_nan)
    is_bad = any_nan(x) or any_nan(y) or tf.math.reduce_any(y<0)
    return tf.logical_not(is_bad)

def get_window_index_matrix(H):
    I_heart = tf.range(-H['window_size'], 0) + 1
    I_resp = I_heart * data_util.RESP_SCALE
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

def read_sigs_flac(rec_id):
    return tf.py_function(partial(data_util.read_sigs), [rec_id], 'float32')

def get_windows(window_size, window_index_matrix, chunk_path, sig_indices, window_indices, baseline, gain):
    sigs = tf.io.read_file(chunk_path)
    sigs = tf.io.parse_tensor(sigs, out_type='int16')
    sigs.set_shape((prepare_data.CHUNK_SIZE, None))
    sigs = tf.concat([tf.zeros(shape=[prepare_data.CHUNK_SIZE, 1], dtype='int16'), sigs], axis=1)
    sigs = tf.gather(sigs, tf.cast(sig_indices, 'int32'), axis=1)
    sigs.set_shape((prepare_data.CHUNK_SIZE, len(S)))
    sigs = tf.cast(tf.cast(sigs, 'int32') - baseline, 'float32') / gain
    offsets = get_offsets(window_indices * window_size)
    windows = tf.gather_nd(sigs, window_index_matrix + offsets)
    x, y = windows[:,:,:-1], windows[:,:,-1]
    y = tf.concat([
        tf.reduce_max(y, axis=1, keepdims=True), 
        tf.reduce_min(y, axis=1, keepdims=True)
    ], axis=-1)
    data = tf.data.Dataset.from_tensor_slices((x, y))
    return data

def build(H, epoch):
    n = len(epoch['chunk_paths'])
    data = tf.data.Dataset.from_tensor_slices((
        tf.constant(epoch['chunk_paths'],    dtype='string',  shape=[n]),
        tf.constant(epoch['sig_indices'],    dtype='int8',    shape=(n, len(S))),
        tf.constant(epoch['window_indices'], dtype='int32',   shape=(n, H['windows_per_chunk'])),
        tf.constant(epoch['baselines'],      dtype='int32',   shape=(n, len(S))),
        tf.constant(epoch['gains'],          dtype='float32', shape=(n, len(S))),
    ))
    window_index_matrix = get_window_index_matrix(H)
    data = data.interleave(
        partial(get_windows, H['window_size'], window_index_matrix), 
        block_length=1, 
        cycle_length=H['batch_buffer_size'] * H['batch_size'],
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    buffer_size = H['batch_buffer_size'] * H['batch_size'] * H['windows_per_chunk']
    data = data.filter(filter_datum).shuffle(buffer_size).batch(H['batch_size'])
    return data
