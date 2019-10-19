import os
from functools import partial, reduce
import json
import pandas
import shutil
import numpy
import tensorflow as tf
from scipy import signal

import flacdb, prepare_data

RESP_SCALE = 5
HOME = '/sailhome/kuprel/blood-pressure/'


def filter_has_sigs(H, metadata):
    if type(metadata['sig_name'][0]) is str:
        metadata['sig_name'] = metadata['sig_name'].apply(eval)
    reducer = all if H['require_all_sigs'] else any
    are_there = lambda i: reducer(j in i for j in H['input_sigs_train'])
    filtered = metadata[metadata['sig_name'].apply(are_there)]
    return filtered

def load_hypes(model_id):
    src = HOME + 'hypes.json'
    dst = HOME + 'hypes/' + model_id + '.json'
    shutil.copy(src, dst)
    with open(dst) as f:
        H = json.load(f)
    return H

def load_metadata(H):
    metadata = pandas.read_csv('/scr-ssd/mimic/metadata.csv')
    metadata = prepare_data.filter_downloaded(metadata)
    metadata = metadata[metadata['sig_len'] > prepare_data.CHUNK_SIZE]
    metadata = filter_has_sigs(H, metadata)
    if H['use_only_matched']:
        metadata = metadata[metadata['subject_id'] != -1]
    return metadata
    
def load_sig_data(H, metadata):
    data = pandas.read_csv('/scr-ssd/mimic/sig_data.csv')
    data = data[data['sig_name'].isin(H['input_sigs_train'] + H['output_sigs'])]
    data.drop_duplicates(['rec_id', 'segment', 'sig_name'], inplace=True)
    data.set_index(['rec_id', 'segment', 'sig_name'], verify_integrity=True, inplace=True)
    data.sort_index(inplace=True)
    data = data[['sig_index', 'baseline', 'adc_gain']].unstack(level=-1)
    data['sig_index'] = data['sig_index'].fillna(-1).astype('int8') + 1
    data['baseline'] = data['baseline'].fillna(0).astype('int32')
    data['adc_gain'] = data['adc_gain'].fillna(0).astype('float64')
    data = data.reindex(metadata.index)
    data.loc[:, 'sig_len'] = metadata['sig_len']
    return data

def load_partition(H, metadata):
    test_subject_ids = open(HOME + 'test_subject_ids.txt').readlines()
    test_subject_ids = [int(i.strip()) for i in test_subject_ids]
    partition = {'validation': metadata['subject_id'].isin(test_subject_ids)}
    partition['train'] = ~partition['validation']
    all_there = lambda i: all(j in i for j in H['input_sigs_validation'])
    partition['validation'] &= metadata['sig_name'].apply(all_there)
    return partition

def calculate_chunks_per_record(H, rec_count, is_validation):
    windows_per_step = H['batch_size']
    steps_per_epoch = H['validation_steps'] if is_validation else H['steps_per_epoch']
    window_count = windows_per_step * steps_per_epoch * H['epochs']
    chunk_count = window_count / H['windows_per_chunk']
    chunk_count *= 7
    chunks_per_record = max(1, round(chunk_count / rec_count))
    return chunks_per_record

def sample_segments(replace, n, data):
    m = data.shape[0]
    if not replace and n > m:
        data = data.iloc[[i for i in range(m) for j in range(n // m + 1)]]
    data = data.sample(n=n, replace=replace)
    return data

def get_chunk_paths(data):
    paths = data.reset_index()[['rec_id', 'segment', 'chunk_id']].values
    rec_ids = paths[:, 0].astype('a7')
    segs = numpy.char.zfill(paths[:, 1].astype('a4'), 4)
    chunk_ids = numpy.char.zfill(paths[:, 2].astype('a4'), 4)
    root = str.encode(prepare_data.ROOT_SERIAL)
    paths = [root, rec_ids, b'_', segs, b'_', chunk_ids, b'.tfrec']
    paths = reduce(numpy.char.add, paths).astype(str)
    return paths

def sample_data(H, data, is_validation=False):
    data.at[:, 'chunk_count'] = [prepare_data.get_chunk_count(i) for i in data['sig_len']]
    rec_count = len(set(data.reset_index()['rec_id']))
    chunks_per_record = calculate_chunks_per_record(H, rec_count, is_validation)
    sample_segs = partial(sample_segments, H['sample_with_replacement'], chunks_per_record)
    data = data.groupby(level=0).apply(sample_segs).droplevel(0)
    data.at[:, 'chunk_index'] = [i for j in range(rec_count) for i in range(chunks_per_record)]
    data = data.reset_index().set_index(['rec_id', 'segment', 'chunk_index'], verify_integrity=True)
    data.sort_index(inplace=True)
    if H['sample_with_replacement']:
        data.at[:, 'chunk_id'] = [numpy.random.randint(i) for i in data['chunk_count']]
    else:
        data.at[:, 'chunk_id'] = range(data.shape[0])
        data['chunk_id'] %= data['chunk_count']
    
    data.at[:, 'chunk_path'] = get_chunk_paths(data)
    I, J = range(data.shape[0]), range(H['windows_per_chunk'])
    data = data.iloc[[i for i in I for j in J]]
    data.at[:, 'window_index'] = [j for i in I for j in J]
    data.at[:, 'window_id'] = numpy.random.randint(
        low = H['window_size'] * RESP_SCALE,
        high = prepare_data.CHUNK_SIZE,
        size = data.shape[0]
    )
    data = data.reset_index().set_index(['rec_id', 'chunk_index', 'window_index'], verify_integrity=True)
    data.sort_index(inplace=True)
    return data

def dataframe_to_tensors(H, data):
    window_indices = data['window_id'].unstack(-1).values
    data = data.loc[(slice(None), slice(None), 0), :]
    S = H['input_sigs_train'] + H['output_sigs']
    n = data.shape[0]
    chunk_paths = data['chunk_path'].values
    sig_indices = data['sig_index'][S].values
    baselines   = data['baseline'][S].values
    gains       = data['adc_gain'][S].values
    I = numpy.random.permutation(n)
    tensors = (
        tf.constant(chunk_paths[I],    dtype='string',  shape=[n]),
        tf.constant(sig_indices[I],    dtype='int8',    shape=(n, len(S))),
        tf.constant(window_indices[I], dtype='int32',   shape=(n, H['windows_per_chunk'])),
        tf.constant(baselines[I],      dtype='int32',   shape=(n, len(S))),
        tf.constant(gains[I],          dtype='float32', shape=(n, len(S))),
    )
    return tensors

def describe_data_size(metadata):
    if type(metadata['sig_name'][0]) is str:
        metadata['sig_name'] = metadata['sig_name'].apply(eval)
    paths = [flacdb.get_rec_path(i) for i in metadata.index]
    paths = [i+j for i in paths for j in ['.hea', '_x.hea', '_x.flac']]
    size = sum(os.path.getsize(i) for i in paths)
    sig_counts = metadata['sig_name'].apply(len)
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    sample_counts = sig_counts * chunk_counts * prepare_data.CHUNK_SIZE
    years = sample_counts.sum() / (125 * 60 * 60 * 24 * 365)
    print(round(size * 1e-9), 'GB, ', int(round(years)), 'years, ', len(metadata), 'record segments')
    
def calculate_training_speed(H, batch_count, seconds_to_train):
    samples_per_day = 125 * 60 * 60 * 24
    hours_to_train = seconds_to_train / 3600
    samples_trained = batch_count * H['batch_size'] * H['window_size'] * len(H['input_sigs_train'])
    days_per_hour = samples_trained / samples_per_day / hours_to_train
    print(round(days_per_hour, ndigits=1), 'days per hour')