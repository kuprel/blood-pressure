import os
from functools import partial, reduce
import json
import pandas
import zlib
import shutil
import numpy
from scipy import signal
import tensorflow as tf

import flacdb
import prepare_data

RESP_SCALE = 5
HOME = '/sailhome/kuprel/blood-pressure/'

TENSOR_DTYPES = {
    'chunk_name':  'string',
    'sig_index':   'int8',
    'window_ids':  'uint16',
    'baseline':    'int16',
    'adc_gain':    'float32',
    'gender':      'int8',
    'age':         'int8',
    'height':      'int8',
    'weight':      'float32',
    'race':        'int8',
    'died':        'int8',
    'diagnosis':   'int8',
    'rec_id':      'int32',
    'segment':     'int16',
    'chunk_count': 'uint8'
}

clinic_file = lambda i: '/scr1/mimic/clinic/{}.csv'.format(i.upper())


def load_hypes(model_id='tmp'):
    src = HOME + 'hypes.json'
    dst = HOME + 'hypes/' + model_id + '.json'
    shutil.copy(src, dst)
    with open(dst) as f:
        H = json.load(f)
    return H


def load_diagnosis(codes, metadata):
    diagnosis = pandas.read_csv(clinic_file('diagnoses_icd'))
    new_names = {'SUBJECT_ID': 'subject_id', 'HADM_ID': 'hadm_id'}
    new_names['ICD9_CODE'] = 'code'
    diagnosis = diagnosis[new_names].rename(columns=new_names)
    diagnosis.loc[~diagnosis['code'].isin(codes), 'code'] = 'other'
    diagnosis.drop_duplicates(inplace=True)
    diagnosis.set_index(['subject_id', 'hadm_id', 'code'], inplace=True)
    diagnosis.sort_index(inplace=True)
    diagnosis.at[:, 'present'] = True
    diagnosis = diagnosis.unstack(fill_value=False)['present'].astype('bool')
    
    index = metadata.reset_index()[diagnosis.index.names]
    index = pandas.MultiIndex.from_frame(index)
    diagnosis = diagnosis.reindex(index).reset_index()
    frames = [diagnosis, metadata.reset_index()[['rec_id', 'segment']]]
    diagnosis = pandas.concat(frames, sort=False, axis=1)

    diagnosis = diagnosis.set_index(['subject_id', 'rec_id', 'segment'])
    diagnosis = diagnosis.sort_index()
    
    is_negative_always = ~diagnosis.drop(columns='hadm_id').any(level=0)
    is_diagnosed_always = (diagnosis['hadm_id'] > 0).all(level=0)
    
    is_negative = is_negative_always[is_diagnosed_always]
    diagnosis = diagnosis.drop(columns='hadm_id')
    bool_to_int = {True: 1, False: 0, numpy.nan: 0}
    diagnosis = diagnosis.replace(bool_to_int)
    diagnosis.loc[is_negative.index] -= is_negative.replace(bool_to_int)
    diagnosis = diagnosis.reset_index()
    diagnosis = diagnosis.set_index(metadata.index.names)
    diagnosis = diagnosis.reindex(metadata.index)
    return diagnosis


def load_test_subject_ids():
    test_subject_ids = open(HOME + 'test_subject_ids.txt').readlines()
    test_subject_ids = [int(i.strip()) for i in test_subject_ids]
    return test_subject_ids


def check_partition(train, val):
    mutually_exclusive = not (train.any(level=0) & val.any(level=0)).any()
    collectively_exhaustive = (train.all(level=0) | val.any(level=0)).all()
    assert(mutually_exclusive and collectively_exhaustive)


def load_partition(H, sig_data, metadata):
    test_subject_ids = load_test_subject_ids()
    partition = {'validation': metadata['subject_id'].isin(test_subject_ids)}
    partition['train'] = ~partition['validation']
    S = [s for s in H['input_sigs_validation'] if s not in H['output_sigs']]
    has_validation_sigs = (sig_data['sig_index'][S] > 0).all(axis=1)
    partition['validation'] &= has_validation_sigs
    partition['train'] |= ~has_validation_sigs.any(level=0)
    check_partition(partition['train'], partition['validation'])
    return partition    


def load_sig_data(sig_names):
    index = ['rec_id', 'segment', 'sig_name']
    columns = index + ['sig_index', 'baseline', 'adc_gain']
    sig_data = pandas.read_hdf('/scr-ssd/mimic/sig_data.hdf', columns=columns)
    sig_data = sig_data[sig_data['sig_name'].isin(sig_names)]
    sig_data.drop_duplicates(index, inplace=True)
    sig_data = sig_data.astype({'sig_name': str})
    dtypes = sig_data.dtypes
    sig_data.set_index(index, inplace=True)
    sig_data = sig_data.loc[(slice(None), slice(None), sig_names), :]
    sig_data.at[:, 'sig_index'] += 1
    sig_data = sig_data.unstack(fill_value=0)
    sig_data = sig_data.astype({(k, s): dtypes[k] for k, s in sig_data})
    return sig_data


def load_metadata():
    metadata = pandas.read_hdf('/scr-ssd/mimic/metadata.hdf')
    metadata = metadata[metadata['sig_len'] > prepare_data.CHUNK_SIZE]
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    metadata.at[:, 'chunk_count'] = chunk_counts.astype('uint8')
    index = (metadata.index & prepare_data.get_downloaded()).sort_values()
    metadata = metadata.reindex(index)
    return metadata


def calculate_chunks_per_record(H, rec_count):
    windows_per_step = H['batch_size']
    window_count = windows_per_step * H['steps_per_epoch'] * H['epochs']
    chunk_count = window_count / H['windows_per_chunk']
    chunk_count *= 1.2
    chunks_per_record = max(1, round(chunk_count / rec_count))
    return chunks_per_record


def sample_segments(replace, n, data):
    m = data.shape[0]
    if not replace and n > m:
        data = data.iloc[[i for i in range(m) for j in range(n // m + 1)]]
    data = data.sample(n=n, replace=replace)
    return data


def get_chunk_names(data):
    names = data.reset_index()[['rec_id', 'segment', 'chunk_id']].values
    rec_ids = names[:, 0].astype('a7')
    segs = numpy.char.zfill(names[:, 1].astype('a4'), 4)
    chunk_ids = numpy.char.zfill(names[:, 2].astype('a4'), 4)
    names = [rec_ids, b'_', segs, b'_', chunk_ids]
    names = reduce(numpy.char.add, names)
    return names


def sample_training_data(H, chunk_counts):
    rec_count = len(set(chunk_counts.reset_index()['rec_id']))
    chunks_per_record = calculate_chunks_per_record(H, rec_count)
    replace = H['sample_with_replacement']
    sample_segs = partial(sample_segments, replace, chunks_per_record)
    sample = chunk_counts.groupby(level=0).apply(sample_segs).droplevel(0)
    I, J = range(chunks_per_record), range(rec_count)
    chunk_index = [i for j in J for i in I]
    sample.at[:, 'chunk_index'] = numpy.array(chunk_index, dtype='int32')
    new_index = ['rec_id', 'segment', 'chunk_index']
    sample = sample.reset_index().set_index(new_index)
    if replace:
        chunk_ids = [numpy.random.randint(i) for i in sample['chunk_count']]
        sample.at[:, 'chunk_id'] = numpy.array(chunk_ids, dtype='uint8')
    else:
        sample.at[:, 'chunk_id'] = range(sample.shape[0])
        sample['chunk_id'] %= sample['chunk_count']
        sample.at[:, 'chunk_id'] = sample['chunk_id'].astype('uint8')
    sample.at[:, 'chunk_name'] = get_chunk_names(sample)
    return sample


def dataframes_to_tensors(H, data):
    train, val = data['partition']['train'], data['partition']['validation']
    tensors = {}
    tensors['train'] = dataframes_to_tensors_train(
        H,
        data['sample'], 
        data['sig_data'][train], 
        data['metadata'][train],
        data['diagnosis'][train]
    )
    tensors['validation'] = dataframes_to_tensors_validation(
        H,
        data['sig_data'][val], 
        data['metadata'][val],
        data['diagnosis'][val]
    )
    for part in ['train', 'validation']:
        for k in tensors[part]:
            dtype = TENSOR_DTYPES[k]
            tensors[part][k] = tf.constant(tensors[part][k], dtype=dtype)
    return tensors


def dataframes_to_tensors_train(H, sample, sig_data, metadata, diagnosis):
    tensors = {}
    tensors['window_ids'] = numpy.random.randint(
        low = H['window_size'] * RESP_SCALE,
        high = prepare_data.CHUNK_SIZE,
        size = (sample.shape[0], H['windows_per_chunk']),
        dtype = 'uint16'
    )
    tensors['chunk_name'] = sample['chunk_name']
    sig_data = sig_data.reindex(sample.index)
    metadata = metadata.reindex(sample.index)
    diagnosis = diagnosis.reindex(sample.index)
    clinic_tensors = clinic_data_to_tensors(H, sig_data, metadata, diagnosis)
    tensors = {**tensors, **clinic_tensors}
    return tensors

def dataframes_to_tensors_validation(H, sig_data, metadata, diagnosis):
    tensors = {}
    tensors['rec_id'] = metadata.reset_index()['rec_id']
    tensors['segment'] = metadata.reset_index()['segment']
    tensors['chunk_count'] = metadata['chunk_count']
    clinic_tensors = clinic_data_to_tensors(H, sig_data, metadata, diagnosis)
    tensors = {**tensors, **clinic_tensors}
    return tensors


def clinic_data_to_tensors(H, sig_data, metadata, diagnosis):
    assert((sig_data.index == metadata.index).all())
    assert((metadata.index == diagnosis.index).all())
    tensors = {}
    tensors['diagnosis'] = diagnosis[H['icd_codes']].astype('int8').values
    S = H['input_sigs'] + H['output_sigs']
    gender = metadata['gender'].astype(object).replace({'M': 1, 'F': -1})
    tensors['gender'] = gender.fillna(0).astype('int8')
    race = ['white', 'black', 'hispanic', 'asian']
    race = metadata['ethnicity'].map(race.index)
    tensors['race'] = race.astype(object).fillna(-1).astype('int8') + 1
    died = metadata['death_time'].notna()
    tensors['died'] = died.astype('int8').replace({0: -1})
    tensors['died'][metadata['hadm_id'] == -1] = 0
    for k in ['sig_index', 'baseline', 'adc_gain']:
        tensors[k] = sig_data[k][S].values
    for k in ['age', 'weight', 'height']:
        tensors[k] = metadata[k].values
    return tensors


def write_tensors(data, file_suffix):
    for k in data:
        file = TENSORS_ROOT + k + '_' + file_suffix + '.tensor'
        x = tf.io.serialize_tensor(data[k])
        z = zlib.compress(x, level=6)
        tf.io.write_file(file, z)

        
def read_tensors(epochs, part=None):
    data = {}
    parts = ['train', 'validation'] if part is None else [part]
    for part in parts:
        data[part] = {}
        for i in TENSOR_INFO:
            file_suffix = str(epochs) + '_' + part
            file = TENSORS_ROOT + i['name'] + '_' + file_suffix + '.tensor'
            z = tf.io.read_file(file)
            x = tf.io.decode_compressed(z, compression_type='ZLIB')
            data[part][i['name']] = tf.io.parse_tensor(x, out_type=i['dtype'])
    return data

        
def run(H):
    sig_data = load_sig_data(H['input_sigs'] + H['output_sigs'])
    metadata = load_metadata()
    index = metadata.index & sig_data.index 
    index &= prepare_data.get_downloaded() 
    sig_data = sig_data.reindex(index)
    metadata = metadata.reindex(index)
    diagnosis = load_diagnosis(H['icd_codes'], metadata)
    
    partition = load_partition(H, sig_data, metadata)
    describe_data_size(H, sig_data, metadata)
    chunk_count = metadata[partition['train']][['chunk_count']]
    sample = sample_training_data(H, chunk_count)
    data = {
        'sig_data': sig_data,
        'metadata': metadata,
        'partition': partition,
        'diagnosis': diagnosis,
        'sample': sample
    }
    return data


def save(path, data):
    if os.path.isfile(path):
        os.remove(path)
    data['sig_data'].to_hdf(path, 'sig_data')
    data['metadata'].to_hdf(path, 'metadata', format='table')
    data['partition']['train'].to_hdf(path, 'partition_train')
    data['partition']['validation'].to_hdf(path, 'partition_validation')
    data['sample'].to_hdf(path, 'sample')


def load(H, path):
    data = {}
    data['sig_data'] = pandas.read_hdf(path, 'sig_data')
    data['metadata'] = pandas.read_hdf(path, 'metadata')
    data['diagnosis'] = load_diagnosis(H['icd_codes'], data['metadata'])
    data['partition'] = {}
    data['partition']['train'] = pandas.read_hdf(path, 'partition_train')
    data['partition']['validation'] = pandas.read_hdf(path, 'partition_validation')
    data['sample'] = pandas.read_hdf(path, 'sample')
    return data

        
def describe_data_size(H, sig_data, metadata):
    sig_counts = (sig_data['sig_index'][H['input_sigs']] > 0).sum(1)
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    sample_counts = sig_counts * chunk_counts * prepare_data.CHUNK_SIZE
    years = sample_counts.sum() / (125 * 60 * 60 * 24 * 365)
    print(int(round(years)), 'years, ', len(metadata), 'record segments')

    
def partition_subject_ids():
    metadata = pandas.read_hdf('/scr-ssd/mimic/metadata.hdf')
    subject_ids = metadata['subject_id'].unique()
    numpy.random.shuffle(subject_ids)
    i = round(0.2*len(subject_ids))
    with open(HOME + 'test_subject_ids.txt', 'w') as f:
        f.write('\n'.join(subject_ids[:i].astype('str')))


def calculate_training_speed(H, batch_count, seconds_to_train):
    samples_per_day = 125 * 60 * 60 * 24
    hours_to_train = seconds_to_train / 3600
    samples_trained = batch_count * H['batch_size'] * H['window_size']
    samples_trained *= len(H['input_sigs'])
    days_per_hour = samples_trained / samples_per_day / hours_to_train
    print(round(days_per_hour, ndigits=1), 'days per hour')