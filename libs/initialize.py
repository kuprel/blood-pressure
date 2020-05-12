import os
from functools import partial, reduce
import json
import pandas
import shutil
import numpy
import tensorflow as tf
import prepare_data
import load_diagnosis

# RESP_SCALE = 5
RESP_SCALE = 1
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
    'seg_id':     'int16',
    'chunk_count': 'uint16'
}

def load_hypes(model_id='tmp'):
    src = HOME + 'hypes.json'
    dst = HOME + 'hypes/' + model_id + '.json'
    shutil.copy(src, dst)
    with open(dst) as f:
        H = json.load(f)
    src = HOME + 'diagnosis_codes.csv'
    dst = HOME + 'diagnosis_codes/' + model_id + '.csv'
    shutil.copy(src, dst)
    return H


def partition_subject_ids():
    metadata = pandas.read_hdf('/scr-ssd/mimic/metadata.hdf')
    subject_ids = metadata['subject_id'].unique()
    numpy.random.shuffle(subject_ids)
    i = len(subject_ids) // 6
    with open('/scr-ssd/mimic/test_subject_ids.txt', 'w') as f:
        f.write('\n'.join(subject_ids[:i].astype('str')))

def load_test_subject_ids():
    test_subject_ids = open('/scr-ssd/mimic/test_subject_ids.txt').readlines()
    test_subject_ids = [int(i.strip()) for i in test_subject_ids]
    return test_subject_ids


def check_partition(train, val):
    mutually_exclusive = not (train.any(level=0) & val.any(level=0)).any()
    collectively_exhaustive = (train.all(level=0) | val.any(level=0)).all()
    assert(mutually_exclusive and collectively_exhaustive)


def load_partition(val_sigs, sig_data):
    subject_ids = sig_data.index.to_frame()['subject_id']
    test_subject_ids = load_test_subject_ids()
    partition = {'validation': subject_ids.isin(test_subject_ids)}
    partition['train'] = ~partition['validation']
    has_validation_sigs = (sig_data['sig_index'][val_sigs] > 0).all(axis=1)
    partition['validation'] &= has_validation_sigs
    partition['train'] |= ~has_validation_sigs.any(level=0)
    check_partition(partition['train'], partition['validation'])
    return partition 

def get_partition(val_sigs, sig_data):
    subject_ids = sig_data.index.to_frame()['subject_id']
    unique_ids = subject_ids.unique()
    test_subject_ids = numpy.random.permutation(unique_ids)[:len(unique_ids)//5]
    partition = {'validation': subject_ids.isin(test_subject_ids)}
    partition['train'] = ~partition['validation']
    has_validation_sigs = (sig_data['sig_index'][val_sigs] > 0).all(axis=1)
    partition['validation'] &= has_validation_sigs
    partition['train'] |= ~has_validation_sigs.any(level=0)
    check_partition(partition['train'], partition['validation'])
    return partition 


def load_initial_data(load_path=None, save_path=None):
    if load_path is not None:
        metadata = pandas.read_pickle(load_path + 'metadata.pkl')
        sig_data = pandas.read_pickle(load_path + 'sig_data.pkl')
        diagnosis = pandas.read_pickle(load_path + 'diagnosis.pkl')
        return sig_data, metadata, diagnosis
    metadata = pandas.read_hdf('/scr-ssd/mimic/metadata.hdf')
    metadata = metadata.reindex(metadata.index & prepare_data.get_serialized())
    metadata = metadata[metadata['sig_len'] > prepare_data.CHUNK_SKIP_SIZE]
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    metadata.at[:, 'chunk_count'] = chunk_counts.astype('uint16')
    missing = metadata['subject_id'] == -1
    fake_ids = -metadata.loc[missing].index.get_level_values(0)
    metadata.at[missing, 'subject_id'] = fake_ids
    subject_ids = metadata['subject_id']
    metadata = metadata.reset_index()
    metadata.set_index(['subject_id', 'rec_id', 'seg_id'], inplace=True)
    metadata.sort_index(inplace=True)
    index_names = ['rec_id', 'seg_id', 'sig_name']
    columns = index_names + ['sig_index', 'baseline', 'adc_gain']
    sig_data = pandas.read_hdf('/scr-ssd/mimic/sig_data.hdf', columns=columns)
    sig_data.drop_duplicates(index_names, inplace=True)
    sig_data = sig_data.astype({'sig_name': str})
    dtypes = sig_data.dtypes
    sig_data.set_index(index_names, inplace=True)
    sig_data.at[:, 'sig_index'] += 1
    sig_data.at[:, 'subject_id'] = subject_ids
    sig_data.fillna({'subject_id': -1}, inplace=True)
    sig_data = sig_data.astype({'subject_id': 'int32'})
    sig_data.reset_index(inplace=True)
    sig_data.set_index(['subject_id'] + index_names, inplace=True)
    sig_data = sig_data.unstack(fill_value=0)
    sig_data = sig_data.astype({(k, s): dtypes[k] for k, s in sig_data})
    index = (metadata.index & sig_data.index).sort_values()
    metadata = metadata.reindex(index)
    sig_data = sig_data.reindex(index)
    diagnosis = load_diagnosis.load_grouped()
    if save_path is not None:
        sig_data.to_pickle(save_path + 'sig_data.pkl')
        metadata.to_pickle(save_path + 'metadata.pkl')
        diagnosis.to_pickle(save_path + 'diagnosis.pkl')
    return sig_data, metadata, diagnosis


def get_row_lengths(metadata):
    row_lengths = metadata.groupby(level=[0, 1]).apply(len)
    row_lengths = row_lengths.groupby(level=0).apply(lambda i: i.values)
    row_lengths = [
        row_lengths.apply(len).values,
        numpy.concatenate(row_lengths.values)
    ]
    return row_lengths


def dataframe_to_ragged_tensor(row_lengths, dataframe, name, nested):
    values = dataframe.values.astype(TENSOR_DTYPES[name])
    if nested:
        t = tf.RaggedTensor.from_nested_row_lengths(values, row_lengths)  
    else:
        t = tf.RaggedTensor.from_row_lengths(values, row_lengths[0])
        t = tf.expand_dims(t, axis=1)
    return t


def to_tensors(H, metadata, sig_data, diagnosis, row_lengths):
    
    to_ragged_tensor = partial(dataframe_to_ragged_tensor, row_lengths)
    
    assert((metadata.index == sig_data.index).all())
    assert((metadata.index == diagnosis.index).all())
    metadata = metadata.reset_index()
    sig_data = sig_data.reset_index()
    diagnosis = diagnosis.reset_index()
    
    tensors = {}
    
    for k in ['seg_id', 'chunk_count']:
        tensors[k] = to_ragged_tensor(metadata[k], k, nested=True)
    
    S = H['input_sigs']
    for k in ['sig_index', 'adc_gain', 'baseline']:
        tensors[k] = to_ragged_tensor(sig_data[k][S], k, nested=True)
    
    metadata = metadata.drop_duplicates(['subject_id', 'rec_id'])
    diagnosis = diagnosis.drop_duplicates(['subject_id', 'rec_id'])
    
    diagnosis = diagnosis.drop(columns=['subject_id', 'rec_id', 'seg_id'])
    
    k = 'diagnosis'
    tensors[k] = to_ragged_tensor(diagnosis, k, nested=False)
    
    for k in ['rec_id', 'height', 'weight', 'age']:
        tensors[k] = to_ragged_tensor(metadata[k], k, nested=False)
    
    return tensors


def describe_data_size(H, sig_data, metadata):
    sig_counts = (sig_data['sig_index'][H['input_sigs']] > 0).sum(1)
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    sample_counts = sig_counts * chunk_counts * prepare_data.CHUNK_SIZE
    years = sample_counts.sum() / (125 * 60 * 60 * 24 * 365)
    print(int(round(years)), 'years, ', len(metadata), 'record segments')

        
def run(H, parts, load_path=None, save_path=None):
    sig_data, metadata, diagnosis = load_initial_data(load_path, save_path)
    diagnosis = load_diagnosis.augment(diagnosis, metadata)
    diagnosis = load_diagnosis.fix(diagnosis)
    priors = (diagnosis == 1).sum() / (diagnosis != 0).sum()
    diagnosis = load_diagnosis.conform(diagnosis, metadata)
#     partition = load_partition(H['input_sigs_validation'], sig_data)
    partition = get_partition(H['input_sigs_validation'], sig_data)
    
    tensors = {}
    for part in parts:
        I = partition[part]
        row_lengths = get_row_lengths(metadata[I])
        args = [metadata[I], sig_data[I], diagnosis[I], row_lengths]
        tensors[part] = to_tensors(H, *args)
        
    return tensors, metadata, priors 