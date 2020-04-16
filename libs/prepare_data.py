import pandas
import numpy
import tensorflow as tf
import os
import multiprocessing

import flacdb


ROOT_SERIAL = '/scr-ssd/mimic/waveforms/'
# CHUNK_SIZE = 2**16
# MAX_CHUNK_COUNT = 16
CHUNK_SIZE = 2**12
CHUNK_SKIP_SIZE = 2**15
# MAX_CHUNK_COUNT = 1

def get_chunk_path(rec, i):
    return ROOT_SERIAL + rec.record_name + '/' + str(i).zfill(4) + '.tfrec'

# def get_chunk_count(sig_len):
#     return min(MAX_CHUNK_COUNT, sig_len // CHUNK_SIZE)

def get_chunk_count(sig_len):
    return max(1, sig_len // CHUNK_SKIP_SIZE)

def get_chunks(rec):
    chunk_count = get_chunk_count(rec.sig_len)
    di = rec.sig_len // chunk_count
    i0 = CHUNK_SKIP_SIZE // 2
    X = [rec.d_signal[i0+i*di:i0+i*di+CHUNK_SIZE] for i in range(chunk_count)]
    return X

def write_chunks(rec):
    for i, x in enumerate(get_chunks(rec)):
        x = tf.constant(x, dtype='int16')
        x = tf.io.serialize_tensor(x)
        tf.io.write_file(get_chunk_path(rec, i), x)

def check_chunks(rec):
    for i, x in enumerate(get_chunks(rec)):
        x_ = tf.io.read_file(get_chunk_path(rec, i))
        x_ = tf.io.parse_tensor(x_, out_type='int16')
        x_ = x_.numpy()
        assert((x_ == x).all())

def _serialize(rec_seg):
    rec = flacdb.read_record(rec_seg, compute_physical=False)
    assert(rec.sig_len == rec.d_signal.shape[0])
    if not os.path.isdir(ROOT_SERIAL + rec.record_name):
        os.makedirs(ROOT_SERIAL + rec.record_name)
    write_chunks(rec)
    check_chunks(rec)
        
def convert_flac_to_serial(rec_seg):
    try:
        _serialize(rec_seg)
    except:
        print(rec_seg, 'failed')

        
def get_downloaded():
    rec_segs = os.listdir(flacdb.ROOT)
    to_tuple = lambda i: flacdb.rec_seg_to_tuple(i.split('.')[0])
    rec_segs = sorted({to_tuple(i) for i in rec_segs if '.flac' in i})
    return rec_segs
        
    
def get_serialized():
    rec_segs = os.listdir(ROOT_SERIAL)
    rec_segs = sorted({flacdb.rec_seg_to_tuple(i) for i in rec_segs})
    return rec_segs

def filter_serialized(metadata):    
    chunk_counts = {}
    serialized = []
    for chunk_file in os.listdir(ROOT_SERIAL):
        rec_seg = '_'.join(chunk_file.split('_')[:2])
        rec_seg = flacdb.rec_seg_to_tuple(rec_seg)
        if rec_seg in chunk_counts:
            chunk_counts[rec_seg] += 1
        else:
            chunk_counts[rec_seg] = 1
        if rec_seg in metadata.index:
            sig_len = metadata.loc[rec_seg]['sig_len']
            if chunk_counts[rec_seg] == get_chunk_count(sig_len):
                serialized.append(rec_seg)
    filtered = metadata.reindex(serialized)
    return filtered

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    metadata = pandas.read_hdf('/scr-ssd/mimic/metadata.hdf')
    filtered = metadata.reindex(metadata.index & get_downloaded())
#     filtered = filter_downloaded(metadata)
    filtered = filtered[filtered['sig_len'] > CHUNK_SKIP_SIZE]
    print(len(filtered), 'records')
#     serialized = filter_serialized(filtered)
#     filtered = filtered.reindex(set(filtered.index) - set(serialized.index))
#     print(len(filtered), 'left to serialize')
#     for i in filtered.index:
#         _serialize(i)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(convert_flac_to_serial, filtered.index)