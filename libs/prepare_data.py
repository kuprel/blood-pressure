import pandas
import numpy
import tensorflow as tf
import os
import multiprocessing

import flacdb


ROOT_SERIAL = '/scr-ssd/mimic/waveforms/'
CHUNK_SIZE = 2**16
MAX_CHUNK_COUNT = 16

def get_chunk_path(rec, i):
    chunk_id = rec.record_name + '_' + str(i).zfill(4)
    return ROOT_SERIAL + chunk_id + '.tfrec'

def get_chunk_count(sig_len):
    return min(MAX_CHUNK_COUNT, sig_len // CHUNK_SIZE)

def get_chunks(rec):
    chunk_count = get_chunk_count(rec.sig_len)
    di = rec.sig_len // chunk_count
    X = [rec.d_signal[i*di:i*di+CHUNK_SIZE] for i in range(chunk_count)]
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

def convert_flac_to_serial(rec_seg):
    try:
        rec = flacdb.read_record(rec_seg, compute_physical=False)
        assert(rec.sig_len == rec.d_signal.shape[0])
        write_chunks(rec)
        check_chunks(rec)
    except:
        print(rec_seg, 'failed')

def filter_downloaded(metadata):
    rec_segs = os.listdir(flacdb.ROOT)
    rec_segs = [i.split('.')[0] for i in rec_segs if '_x.flac' in i]
    rec_segs = [flacdb.rec_seg_to_tuple(i) for i in rec_segs]
    if metadata.index.names != ['rec_id', 'segment']:
        metadata = metadata.set_index(['rec_id', 'segment'])
    rec_segs = sorted(set(rec_segs).intersection(metadata.index))
    filtered = metadata.reindex(rec_segs)
    return filtered

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
            required_chunk_count = get_chunk_count(metadata.loc[rec_seg]['sig_len'])
            if chunk_counts[rec_seg] == required_chunk_count:
                serialized.append(rec_seg)
    filtered = metadata.reindex(serialized)
    return filtered

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    metadata = pandas.read_csv('/scr-ssd/mimic/metadata.csv')
    filtered = filter_downloaded(metadata)
    filtered = filtered[filtered['sig_len'] > CHUNK_SIZE]
    print(len(filtered), 'records')
    serialized = filter_serialized(filtered)
    filtered = filtered.reindex(set(filtered.index).difference(serialized.index))
    print(len(filtered), 'left to serialize')
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(convert_flac_to_serial, filtered.index)