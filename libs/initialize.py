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

clinic_file = lambda i: '/scr1/mimic/clinic/{}.csv'.format(i.upper())


def load_hypes(model_id='tmp'):
    src = HOME + 'hypes.json'
    dst = HOME + 'hypes/' + model_id + '.json'
    shutil.copy(src, dst)
    with open(dst) as f:
        H = json.load(f)
    return H


def load_diagnoses(codes, metadata):
    diagnoses = pandas.read_csv(clinic_file('diagnoses_icd'))
    new_names = {'HADM_ID': 'hadm_id', 'ICD9_CODE': 'code'}
    data = diagnoses[new_names].rename(columns=new_names)
    data = data[data['code'].isin(codes)]
    data.drop_duplicates(inplace=True)
    data.set_index(['hadm_id', 'code'], inplace=True)
    data.sort_index(inplace=True)
    data.at[:, 'present'] = True
    data = data.unstack(fill_value=False)['present']
    data = pandas.concat([
        data.reindex(metadata['hadm_id']).reset_index(), 
        metadata.reset_index()[['rec_id', 'segment']]], 
        sort=False, 
        verify_integrity=True,
        axis=1
    )
    data.drop(columns='hadm_id', inplace=True)
    data.set_index(['rec_id', 'segment'], inplace=True)
    return data

def load_test_subject_ids():
    test_subject_ids = open(HOME + 'test_subject_ids.txt').readlines()
    test_subject_ids = [int(i.strip()) for i in test_subject_ids]
    return test_subject_ids


def load_partition(H, sig_data, metadata):
    test_subject_ids = load_test_subject_ids()
    partition = {'validation': metadata['subject_id'].isin(test_subject_ids)}
    partition['train'] = ~partition['validation']    
    has_validation_sig = sig_data['sig_index'][H['input_sigs_validation']] > 0
    partition['validation'] &= has_validation_sig.all(axis=1)
    return partition    


def load_sig_data(sig_names):
    index = ['rec_id', 'segment', 'sig_name']
    columns = index + ['sig_index', 'baseline', 'adc_gain']
    sig_data = pandas.read_hdf('/scr-ssd/mimic/sig_data.hdf', columns=columns)
    dtypes = sig_data.dtypes
    sig_data = sig_data[sig_data['sig_name'].isin(sig_names)]
    sig_data.drop_duplicates(index, inplace=True)
    sig_data.set_index(index, inplace=True)
    sig_data = sig_data.loc[(slice(None), slice(None), sig_names), :]
    sig_data.at[:, 'sig_index'] += 1
    sig_data = sig_data.unstack(fill_value=0)
    sig_data = sig_data.astype({(k, s): dtypes[k] for k, s in sig_data})
    return sig_data


def load_data(H):
    sig_data = load_sig_data(H['input_sigs_train'] + H['output_sigs'])
    metadata = load_metadata(H['use_only_matched'])
    metadata = sample_data
    index = metadata.index & sig_data.index & prepare_data.get_downloaded() 
    sig_data = sig_data.reindex(index)
    metadata = metadata.reindex(index)
    partition = load_partition(H, sig_data, metadata)
    diagnoses = load_diagnoses(H['icd_codes'], metadata)
    return sig_data, metadata, partition, diagnoses


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


def sample_data(H, data, part):
    data = data[data['sig_len'] > prepare_data.CHUNK_SIZE]
    chunk_counts = data['sig_len'].apply(prepare_data.get_chunk_count)
    data.at[:, 'chunk_count'] = chunk_counts.astype('uint8')
    rec_count = len(set(data.reset_index()['rec_id']))
    chunks_per_record = calculate_chunks_per_record(H, rec_count, part)
    replace = H['sample_with_replacement']
    sample_segs = partial(sample_segments, replace, chunks_per_record)
    data = data.groupby(level=0).apply(sample_segs).droplevel(0)
    I, J = range(chunks_per_record), range(rec_count)
    chunk_index = [i for j in J for i in I]
    data.at[:, 'chunk_index'] = numpy.array(chunk_index, dtype='int32')
    new_index = ['rec_id', 'segment', 'chunk_index']
    data = data.reset_index().set_index(new_index)
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
    window_index = [j for i in I for j in J]
    data.at[:, 'window_index'] = numpy.array(window_index, dtype='uint32')
    data.at[:, 'window_id'] = numpy.random.randint(
        low = H['window_size'] * RESP_SCALE,
        high = prepare_data.CHUNK_SIZE,
        size = data.shape[0],
        dtype = 'uint16'
    )
    return data


def dataframes_to_tensors(H, sample, sig_data, metadata, diagnoses):
    arrays = {}
    index = sample.index.names + ['window_index']
    sample = sample.reset_index().set_index(index)
    window_ids = sample['window_id'].unstack(-1).values
    sample = sample.loc[(slice(None), slice(None), slice(None), 0), :]
    sig_data = sig_data.reindex(sample.index)
    metadata = metadata.reindex(sample.index)
    diagnoses = diagnoses.reindex(sample.index)
    S = H['input_sigs_train'] + H['output_sigs']
    gender = numpy.stack([metadata['gender'] == i for i in ['M', 'F']], axis=1)
    race = list(metadata['ethnicity'].dtype.categories)
    race = metadata['ethnicity'].map(race.index)
    race = race.astype(object).fillna(-1).astype('int8') + 1
    diagnoses = diagnoses.astype(object).replace({False: -1, True: 1})
    diagnoses = diagnoses.fillna(0).astype('int8')
    tensors = (
        tf.constant(sample['chunk_name'].values,      dtype='string' ),
        tf.constant(sig_data['sig_index'][S].values,  dtype='int8'   ),
        tf.constant(window_ids,                       dtype='uint16' ),
        tf.constant(sig_data['baseline'][S].values,   dtype='int16'  ),
        tf.constant(sig_data['adc_gain'][S].values,   dtype='float32'),
        tf.constant(gender,                           dtype='bool'   ),
        tf.constant(metadata['age'].values,           dtype='int8'   ),
        tf.constant(metadata['height'].values,        dtype='int8'   ),
        tf.constant(metadata['weight'].values,        dtype='float32'),
        tf.constant(race,                             dtype='int8'   ),
        tf.constant(metadata['death_time'].notna(),   dtype='bool'   ),
        tf.constant(diagnoses.values,                 dtype='int8'   )
    )
    return tensors


def describe_data_size(H, sig_data, metadata):
    sig_counts = (sig_data['sig_index'][H['input_sigs_train']] > 0).sum(1)
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    sample_counts = sig_counts * chunk_counts * prepare_data.CHUNK_SIZE
    years = sample_counts.sum() / (125 * 60 * 60 * 24 * 365)
    print(int(round(years)), 'years, ', len(metadata), 'record segments')

    
def partition_subject_ids():
    metadata = pandas.read_csv('/scr-ssd/mimic/metadata_matched.csv')
    subject_ids = metadata['subject_id'].unique()
    numpy.random.shuffle(subject_ids)
    i = round(0.2*len(subject_ids))
    with open('../test_subject_ids.txt', 'w') as f:
        f.write('\n'.join(subject_ids[:i].astype('str')))


def calculate_training_speed(H, batch_count, seconds_to_train):
    samples_per_day = 125 * 60 * 60 * 24
    hours_to_train = seconds_to_train / 3600
    samples_trained = batch_count * H['batch_size'] * H['window_size']
    samples_trained *= len(H['input_sigs_train'])
    days_per_hour = samples_trained / samples_per_day / hours_to_train
    print(round(days_per_hour, ndigits=1), 'days per hour')