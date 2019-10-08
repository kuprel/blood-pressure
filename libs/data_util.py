import os
from functools import partial
import pickle
import numpy
import pandas
from scipy import signal

from Brett import db, prepare_data

RESP_SCALE = 5

def get_rec_path(rec_id):
    if type(rec_id) is tuple:
        rec_id = db.rec_id_to_string(rec_id)
    return '/scr1/mimic/' + 'waveforms/' + rec_id

def read_sigs(rec_id):
    rec_id = rec_id.numpy().decode('utf-8')
    rec = db.read_record(get_rec_path(rec_id), x_only=True)
    cols = [rec.sig_name.index(j) for j in prepare_data.INPUT_SIGS]
    sigs = rec.p_signal[:, cols]
    return sigs

def sample_epoch(H, metadata, sig_data, seed=None):
    shuffled = metadata.sample(frac=1, random_state=seed)
    sig_data = sig_data.reindex(shuffled.index)
    shuffled = shuffled.reset_index()
    S = prepare_data.INPUT_SIGS + prepare_data.OUTPUT_SIGS
    sig_indices = sig_data['sig_index_new'][S].values
    baselines   = sig_data['baseline'][S].values
    gains       = sig_data['adc_gain'][S].values
    chunk_counts = shuffled['sig_len'].apply(prepare_data.get_chunk_count)
    chunk_indices = chunk_counts.apply(numpy.random.randint)
    chunk_coords = zip(shuffled['rec_id_prefix'], shuffled['segment'], chunk_indices)
    chunk_path = '/scr-ssd/mimic/waveforms/{}_{}_{}.tfrec'
    chunk_paths = [
        chunk_path.format(i, str(j).zfill(4), str(k).zfill(4)) 
        for i, j, k in chunk_coords
    ]
    
    window_indices = numpy.random.randint(
        low = RESP_SCALE,
        high = prepare_data.CHUNK_SIZE // H['window_size'],
        size = [len(chunk_indices), H['windows_per_chunk']]
    )
    
    epoch = {
        'chunk_paths': chunk_paths,
        'window_indices': window_indices,
        'sig_indices': sig_indices,
        'baselines': baselines,
        'gains': gains
    }
    
    return epoch 

def smooth_pressure(y, w):
    w = min(w, y.shape[0])
    w -= w % 2 == 0
    return signal.savgol_filter(y, w, 1, axis=0)

def load_sig_data():
    S = prepare_data.INPUT_SIGS + prepare_data.OUTPUT_SIGS
    data = pandas.read_csv('/scr-ssd/mimic/sig_data.csv')
    data = data[data['sig_name'].isin(S)]
    data['sig_present'] = True
    data['sig_index_new'] = 0
    data.drop_duplicates(['rec_id_prefix', 'segment', 'sig_name'], inplace=True)
    data.set_index(['rec_id_prefix', 'segment', 'sig_name'], verify_integrity=True, inplace=True)
    data.sort_index(inplace=True)
    data = data[['sig_present', 'sig_index_new', 'baseline', 'adc_gain']].unstack(level=-1)
    data['sig_present'] = data['sig_present'].fillna(False).astype('bool')
    data['sig_index_new'] = data['sig_index_new'].fillna(0).astype('int8')
    data['baseline'] = data['baseline'].fillna(0).astype('int32')
    data['adc_gain'] = data['adc_gain'].fillna(0).astype('float64')
    for s in S:
        i = data['sig_present', s]
        data.loc[i, ('sig_index_new', s)] = data['sig_index_new'].max(axis=1)[i] + 1
    return data

def filter_downloaded(metadata):
    rec_ids = os.listdir('/scr1/mimic/' + 'waveforms')
    rec_ids = [i.split('.')[0] for i in rec_ids if '_x.flac' in i]
    rec_ids = [db.rec_id_to_tuple(i) for i in rec_ids]
    if metadata.index.names != ['rec_id_prefix', 'segment']:
        metadata = metadata.set_index(['rec_id_prefix', 'segment'])
    rec_ids = sorted(set(rec_ids).intersection(metadata.index))
    filtered = metadata.reindex(rec_ids)
    return filtered

def filter_has_sigs(metadata, reducer):
    if type(metadata['sig_name'][0]) is str:
        metadata['sig_name'] = metadata['sig_name'].apply(eval)
    are_there = lambda i: reducer(j in i for j in prepare_data.INPUT_SIGS)
    filtered = metadata[metadata['sig_name'].apply(are_there)]
    return filtered

def load_initial_data(require_all_sigs=False):
    metadata = pandas.read_csv('/scr-ssd/mimic/' + 'metadata.csv')
    metadata = filter_downloaded(metadata)
    metadata = metadata[metadata['sig_len'] > prepare_data.CHUNK_SIZE]
    metadata = filter_has_sigs(metadata, reducer=all if require_all_sigs else any)
    metadata.sort_index(inplace=True)
    sig_data = load_sig_data()
    sig_data = sig_data.reindex(metadata.index)
    return metadata, sig_data

def describe_data_size(metadata):
    if type(metadata['sig_name'][0]) is str:
        metadata['sig_name'] = metadata['sig_name'].apply(eval)
    paths = [get_rec_path(i) for i in metadata.index]
    paths = [i+j for i in paths for j in ['.hea', '_x.hea', '_x.flac']]
    size = sum(os.path.getsize(i) for i in paths)
    S = set(prepare_data.INPUT_SIGS + prepare_data.OUTPUT_SIGS)
    sig_counts = metadata['sig_name'].apply(S.intersection).apply(len)
    chunk_counts = metadata['sig_len'].apply(prepare_data.get_chunk_count)
    sample_counts = sig_counts * chunk_counts * prepare_data.CHUNK_SIZE
    years = sample_counts.sum() / (125 * 60 * 60 * 24 * 365)
    print(round(size * 1e-9), 'GB, ', int(round(years)), 'years')
    
def calculate_training_speed(H, batch_count, seconds_to_train):
    samples_per_day = 125 * 60 * 60 * 24
    hours_to_train = seconds_to_train / 3600
    samples_trained = batch_count * H['batch_size'] * H['window_size'] * len(prepare_data.INPUT_SIGS)
    days_per_hour = samples_trained / samples_per_day / hours_to_train
    print(round(days_per_hour, ndigits=1), 'days per hour')