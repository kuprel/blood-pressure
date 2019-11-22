import os
from functools import partial, reduce
import json
import pandas
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
    'seg_id':     'int16',
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


def load_initial_data(load_path=None, save_path=False):
    if load_path is not None:
        metadata = pandas.read_pickle(load_path + 'metadata.pkl')
        sig_data = pandas.read_pickle(load_path + 'sig_data.pkl')
        return sig_data, metadata
    metadata = pandas.read_hdf('/scr-ssd/mimic/metadata.hdf')
    metadata = metadata[metadata['sig_len'] > prepare_data.CHUNK_SIZE]
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    metadata.at[:, 'chunk_count'] = chunk_counts.astype('uint8')
    index = (metadata.index & prepare_data.get_serialized()).sort_values()
    metadata = metadata.reindex(index)
    missing = metadata['subject_id'] == -1
    fake_ids = -metadata.loc[missing].index.get_level_values(0)
    metadata.at[missing, 'subject_id'] = fake_ids
    subject_ids = metadata['subject_id']
    metadata = metadata.reset_index()
    metadata.set_index(['subject_id', 'rec_id', 'seg_id'], inplace=True, verify_integrity=True)
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
    sig_data.set_index(['subject_id'] + index_names, inplace=True, verify_integrity=True)
    sig_data = sig_data.unstack(fill_value=0)
    sig_data = sig_data.astype({(k, s): dtypes[k] for k, s in sig_data})
    sig_data = sig_data[sig_data['sig_index']['ABP'] > 0]
    index = (metadata.index & sig_data.index).sort_values()
    metadata = metadata.reindex(index)
    sig_data = sig_data.reindex(index)
    if save_path is not None:
        sig_data.to_pickle(save_path + 'sig_data.pkl')
        metadata.to_pickle(save_path + 'metadata.pkl')
    return sig_data, metadata


def load_diagnosis(codes, metadata):
    diagnosis = pandas.read_csv(clinic_file('diagnoses_icd'))
    new_names = {i.upper(): i for i in ['subject_id', 'hadm_id']}
    new_names['ICD9_CODE'] = 'code'
    diagnosis = diagnosis[new_names].rename(columns=new_names)
    diagnosis.loc[~diagnosis['code'].isin(codes), 'code'] = 'other'
    diagnosis.drop_duplicates(inplace=True)
    diagnosis.set_index(['subject_id', 'hadm_id', 'code'], inplace=True)
    diagnosis.sort_index(inplace=True)
    diagnosis.at[:, 'present'] = True
    diagnosis = diagnosis.unstack(fill_value=False)['present'].astype('bool')
    return diagnosis


def augment_diagnosis(diagnosis, metadata):
    matched_data = metadata.reset_index()
    matched_data = matched_data[matched_data['subject_id'] > 0]
    matched_data.drop_duplicates(diagnosis.index.names, inplace=True)
    matched_data.set_index(diagnosis.index.names, inplace=True)
    matched_data.sort_index(inplace=True)
        
    diagnosis = diagnosis.reindex(matched_data.index)
    
    for i in ['gender', 'race']:
        values = matched_data[i]
        for j in values.dtype.categories:
            diagnosis.at[values.notna(), i + '_' + j] = values == j
    
    thresholds = {'age': 75, 'height': 70, 'weight': 100}
    
    for k in thresholds:
        is_int = numpy.issubdtype(matched_data[k].dtype, numpy.signedinteger)
        is_given = matched_data[k] > 0 if is_int else matched_data[k].notna()
        name = k + '_at_least_' + str(round(thresholds[k]))
        diagnosis.at[is_given, name] = matched_data[k] >= thresholds[k]
    
    died = matched_data['death_time'].notna()
    diagnosis.at[(slice(None), slice(0, None)), 'died'] = died
    
    bool_to_int = {True: 1, False: -1, numpy.nan: 0}
    diagnosis = diagnosis.replace(bool_to_int).astype('int8')
    
    return diagnosis


def fix_diagnosis(diagnosis):
    is_negative_always = (diagnosis == -1).all(level=0)
    is_diagnosed_always = (diagnosis.index.to_frame()['hadm_id'] > 0).all(level=0)
    is_negative = is_negative_always[is_diagnosed_always]
    diagnosis = diagnosis.replace({-1: 0})
    I = is_negative.index
    diagnosis.loc[I] = diagnosis.loc[I].mask(is_negative, -1)
    diagnosis = diagnosis.drop(columns='other')
    return diagnosis.astype('int8')


def conform_diagnosis(diagnosis, metadata):
    unindexed = metadata.reset_index()
    I = pandas.MultiIndex.from_frame(unindexed[diagnosis.index.names])
    diagnosis = diagnosis.reindex(I).fillna(0).astype('int8').reset_index()
    frames = [diagnosis, unindexed[['rec_id', 'seg_id']]]
    diagnosis = pandas.concat(frames, sort=False, axis=1)
    diagnosis = diagnosis.set_index(metadata.index.names)
    diagnosis = diagnosis.drop(columns='hadm_id')
    assert((diagnosis.index == metadata.index).all())
    return diagnosis


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


def get_tensors(H, metadata, sig_data, diagnosis, row_lengths):
    
    to_ragged_tensor = partial(dataframe_to_ragged_tensor, row_lengths)
    
    assert((metadata.index == sig_data.index).all())
    assert((metadata.index == diagnosis.index).all())
    metadata = metadata.reset_index()
    sig_data = sig_data.reset_index()
    diagnosis = diagnosis.reset_index()
    
    tensors = {}
    
    for k in ['seg_id', 'chunk_count']:
        tensors[k] = to_ragged_tensor(metadata[k], k, nested=True)
    
    S = H['input_sigs'] + H['output_sigs']
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

    
def partition_subject_ids():
    metadata = pandas.read_hdf('/scr-ssd/mimic/metadata.hdf')
    subject_ids = metadata['subject_id'].unique()
    numpy.random.shuffle(subject_ids)
    i = round(0.2*len(subject_ids))
    with open('/scr-ssd/mimic/test_subject_ids.txt', 'w') as f:
        f.write('\n'.join(subject_ids[:i].astype('str')))
