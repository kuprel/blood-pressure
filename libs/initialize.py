import os
from functools import partial, reduce
import json
import pandas
import shutil
import numpy
from scipy import signal
import tensorflow as tf

import flacdb, prepare_data

RESP_SCALE = 5
HOME = '/sailhome/kuprel/blood-pressure/'

def load_hypes(model_id='tmp'):
    src = HOME + 'hypes.json'
    dst = HOME + 'hypes/' + model_id + '.json'
    shutil.copy(src, dst)
    with open(dst) as f:
        H = json.load(f)
    return H

def load_partition(H, sig_data, metadata):
    test_subject_ids = open(HOME + 'test_subject_ids.txt').readlines()
    test_subject_ids = [int(i.strip()) for i in test_subject_ids]
    partition = {'validation': metadata['subject_id'].isin(test_subject_ids)}
    partition['train'] = ~partition['validation']    
    has_validation_sig = sig_data['sig_index'][H['input_sigs_validation']] > 0
    partition['validation'] &= has_validation_sig.all(axis=1)
    return partition    

def load_sig_data(H):
    sig_data = pandas.read_hdf('/scr-ssd/mimic/sig_data.hdf')
    dtypes = sig_data.dtypes
    sig_names = H['input_sigs_train'] + H['output_sigs']
    sig_data = sig_data[sig_data['sig_name'].isin(sig_names)]
    sig_data.drop_duplicates(['rec_id', 'segment', 'sig_name'], inplace=True)
    sig_data.set_index(['rec_id', 'segment', 'sig_name'], inplace=True)
    drop_cols = set(sig_data.columns) - {'sig_index', 'baseline', 'adc_gain'}
    sig_data.drop(columns=drop_cols, inplace=True)
    sig_data.at[:, 'sig_index'] += 1
    sig_data = sig_data.unstack(fill_value=0)
    sig_data = sig_data.astype({(k, s): dtypes[k] for k, s in sig_data})
    return sig_data

def load_metadata(H):
    metadata = pandas.read_hdf('/scr-ssd/mimic/metadata.hdf')
    metadata = metadata.set_index(['rec_id', 'segment'])
    metadata = metadata[metadata['sig_len'] > prepare_data.CHUNK_SIZE]
    if H['use_only_matched']:
        metadata = metadata[metadata['subject_id'] != -1]
    return metadata

def load_data(H):
    sig_data = load_sig_data(H)
    metadata = load_metadata(H)
    index = metadata.index & sig_data.index
    index &= prepare_data.get_downloaded() 
    sig_data = sig_data.reindex(index)
    metadata = metadata.reindex(index)
    partition = load_partition(H, sig_data, metadata)
    return sig_data, metadata, partition

def calculate_chunks_per_record(H, rec_count, part):
    windows_per_step = H['batch_size']
    if part == 'train':
        steps_per_epoch = H['steps_per_epoch']
    else:
        steps_per_epoch = H['validation_steps']
    window_count = windows_per_step * steps_per_epoch * H['epochs']
    chunk_count = window_count / H['windows_per_chunk']
    chunk_count *= 1.2 if part == 'train' else 7
    chunks_per_record = max(1, round(chunk_count / rec_count))
    return chunks_per_record

def sample_segments(replace, n, data):
    m = data.shape[0]
    if not replace and n > m:
        data = data.iloc[[i for i in range(m) for j in range(n // m + 1)]]
    data = data.sample(n=n, replace=replace)
    return data

def get_chunk_names(data):
    paths = data.reset_index()[['rec_id', 'segment', 'chunk_id']].values
    rec_ids = paths[:, 0].astype('a7')
    segs = numpy.char.zfill(paths[:, 1].astype('a4'), 4)
    chunk_ids = numpy.char.zfill(paths[:, 2].astype('a4'), 4)
    paths = [rec_ids, b'_', segs, b'_', chunk_ids]
    paths = reduce(numpy.char.add, paths)
    return paths

def sample_data(H, data, metadata, part):
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    metadata.at[:, 'chunk_count'] = chunk_counts.astype('uint8')
    rec_count = len(set(metadata.reset_index()['rec_id']))
    chunks_per_record = calculate_chunks_per_record(H, rec_count, part)
    replace = H['sample_with_replacement']
    sample_segs = partial(sample_segments, replace, chunks_per_record)
    data = data.groupby(level=0).apply(sample_segs).droplevel(0)
    chunk_index = [i for j in range(rec_count) for i in range(chunks_per_record)]
    data.at[:, 'chunk_index'] = numpy.array(chunk_index, dtype='int32')
    data = data.reset_index()
    data = data.set_index(['rec_id', 'segment', 'chunk_index'], verify_integrity=True)
    data.sort_index(inplace=True)
    if H['sample_with_replacement']:
        chunk_ids = [numpy.random.randint(i) for i in data['chunk_count']]
        data.at[:, 'chunk_id'] = numpy.array(chunk_ids, dtype='uint8')
    else:
        data.at[:, 'chunk_id'] = range(data.shape[0])
        data['chunk_id'] %= data['chunk_count']
        data.at[:, 'chunk_id'] = data['chunk_id'].astype('uint8')
    data.at[:, 'chunk_name'] = get_chunk_names(data)
    I, J = range(data.shape[0]), range(H['windows_per_chunk'])
    data = data.iloc[[i for i in I for j in J]]
    data.at[:, 'window_index'] = numpy.array([j for i in I for j in J], dtype='uint32')
    data.at[:, 'window_id'] = numpy.random.randint(
        low = H['window_size'] * RESP_SCALE,
        high = prepare_data.CHUNK_SIZE,
        size = data.shape[0],
        dtype = 'uint16'
    )
    data = data.reset_index()
    data = data.set_index(['rec_id', 'chunk_index', 'window_index'], verify_integrity=True)
    data.sort_index(inplace=True)
    return data

def sample_data_old(H, data, metadata, part):
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    data.at[:, 'chunk_count'] = chunk_counts.astype('uint8')
    rec_count = len(set(data.reset_index()['rec_id']))
    chunks_per_record = calculate_chunks_per_record(H, rec_count, part)
    sample_segs = partial(sample_segments, H['sample_with_replacement'], chunks_per_record)
    data = data.groupby(level=0).apply(sample_segs).droplevel(0)
    chunk_index = [i for j in range(rec_count) for i in range(chunks_per_record)]
    data.at[:, 'chunk_index'] = numpy.array(chunk_index, dtype='int32')
    data = data.reset_index()
    data = data.set_index(['rec_id', 'segment', 'chunk_index'], verify_integrity=True)
    data.sort_index(inplace=True)
    if H['sample_with_replacement']:
        chunk_ids = [numpy.random.randint(i) for i in data['chunk_count']]
        data.at[:, 'chunk_id'] = numpy.array(chunk_ids, dtype='uint8')
    else:
        data.at[:, 'chunk_id'] = range(data.shape[0])
        data['chunk_id'] %= data['chunk_count']
        data.at[:, 'chunk_id'] = data['chunk_id'].astype('uint8')
    data.at[:, 'chunk_name'] = get_chunk_names(data)
    I, J = range(data.shape[0]), range(H['windows_per_chunk'])
    data = data.iloc[[i for i in I for j in J]]
    data.at[:, 'window_index'] = numpy.array([j for i in I for j in J], dtype='uint32')
    data.at[:, 'window_id'] = numpy.random.randint(
        low = H['window_size'] * RESP_SCALE,
        high = prepare_data.CHUNK_SIZE,
        size = data.shape[0],
        dtype = 'uint16'
    )
    data = data.reset_index()
    data = data.set_index(['rec_id', 'chunk_index', 'window_index'], verify_integrity=True)
    data.sort_index(inplace=True)
    return data

def dataframe_to_tensors(H, dataframe):
    arrays = {}
    arrays['window_indices'] = dataframe['window_id'].unstack(-1).values
    dataframe = dataframe.loc[(slice(None), slice(None), 0), :]
    S = H['input_sigs_train'] + H['output_sigs']
    n = dataframe.shape[0]
    arrays['chunk_name'] = dataframe['chunk_name'].values
    arrays['sig_index']  = dataframe['sig_index'][S].values
    arrays['adc_gain']   = dataframe['adc_gain'][S].values
    arrays['baseline']   = dataframe['baseline'][S].values
    assert(arrays['sig_index'].dtype      == 'int8')
    assert(arrays['baseline'].dtype       == 'int16')
    assert(arrays['adc_gain'].dtype       == 'float32')
    assert(arrays['chunk_name'].dtype     == 'a17')
    assert(arrays['window_indices'].dtype == 'uint16')
    tensors = (
        tf.constant(arrays['chunk_name'],     dtype='string' ),
        tf.constant(arrays['sig_index'],      dtype='int8'   ),
        tf.constant(arrays['window_indices'], dtype='uint16' ),
        tf.constant(arrays['baseline'],       dtype='int16'  ),
        tf.constant(arrays['adc_gain'],       dtype='float32'),
    )
    return tensors

def describe_data_size(H, sig_data, metadata):
    sig_counts = (sig_data['sig_index'][H['input_sigs_train']] > 0).sum(1)
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    sample_counts = sig_counts * chunk_counts * prepare_data.CHUNK_SIZE
    years = sample_counts.sum() / (125 * 60 * 60 * 24 * 365)
    print(int(round(years)), 'years, ', len(metadata), 'record segments')
    
def calculate_training_speed(H, batch_count, seconds_to_train):
    samples_per_day = 125 * 60 * 60 * 24
    hours_to_train = seconds_to_train / 3600
    samples_trained = batch_count * H['batch_size'] * H['window_size']
    samples_trained *= len(H['input_sigs_train'])
    days_per_hour = samples_trained / samples_per_day / hours_to_train
    print(round(days_per_hour, ndigits=1), 'days per hour')