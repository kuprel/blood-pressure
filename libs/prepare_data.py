import pandas
import numpy
import math
import tensorflow as tf
import os
import multiprocessing

from Brett import db, data_util

ROOT_FLAC = '/scr1/mimic/waveforms/'
ROOT_SERIAL = '/scr-ssd/mimic/waveforms/'
CHUNK_SIZE = 2**16
MAX_CHUNK_COUNT = 16
INPUT_SIGS = ['II', 'V', 'PLETH', 'RESP']
OUTPUT_SIGS = ['ABP']

def get_chunk_path(rec, i):
    chunk_id = rec.record_name + '_' + str(i).zfill(4)
    return ROOT_SERIAL + chunk_id + '.tfrec'

def get_chunk_count(sig_len):
    return min(MAX_CHUNK_COUNT, math.floor(sig_len / CHUNK_SIZE))

def get_chunks(rec):
    S = INPUT_SIGS + OUTPUT_SIGS
    I = [rec.sig_name.index(i) for i in S if i in rec.sig_name]
    chunk_count = get_chunk_count(rec.sig_len)
    X = rec.d_signal[:chunk_count*CHUNK_SIZE, I]
    X = numpy.split(X, chunk_count)
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

def convert_flac_to_serial(rec_id):
    rec = db.read_record(rec_id, ROOT_FLAC, compute_physical=False)
    assert(rec.sig_len == rec.d_signal.shape[0])
    write_chunks(rec)
    check_chunks(rec)
        
def filter_not_serialized(metadata):    
    chunk_count = {}
    serialized = []
    for chunk_file in os.listdir(ROOT_SERIAL):
        rec_id = '_'.join(chunk_file.split('_')[:2])
        rec_id = db.rec_id_to_tuple(rec_id)
        if rec_id in chunk_count:
            chunk_count[rec_id] += 1
        else:
            chunk_count[rec_id] = 1
        if rec_id in metadata.index:
            required_chunk_count = get_chunk_count(metadata.loc[rec_id]['sig_len'])
            if chunk_count[rec_id] == required_chunk_count:
                serialized.append(rec_id)      
    filtered = metadata.reindex(set(metadata.index).difference(serialized))
    return filtered

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    metadata = pandas.read_csv('/scr1/mimic/metadata.csv')
    filtered = data_util.filter_downloaded(metadata)
    filtered = filtered[filtered['sig_len'] > CHUNK_SIZE]
    print(len(filtered), 'records')
    filtered = filter_not_serialized(filtered)
    print(len(filtered), 'left to serialize')
    data_util.describe_data_size(filtered)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(convert_flac_to_serial, filtered.index)