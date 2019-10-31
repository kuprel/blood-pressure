import pickle
import initialize
import prepare_data
import pandas


H = initialize.load_hypes()
sig_data = initialize.load_sig_data(H['input_sigs_train'] + H['output_sigs'])
metadata = pandas.read_hdf('/scr-ssd/mimic/metadata.hdf')
index = metadata.index & sig_data.index & prepare_data.get_downloaded() 
sig_data = sig_data.reindex(index)
metadata = metadata.reindex(index)
partition = initialize.load_partition(H, sig_data, metadata)
diagnoses = initialize.load_diagnoses(H['icd_codes'], metadata)
initialize.describe_data_size(H, sig_data, metadata)

data = {}
for part in ['train', 'validation']:
    sig_len = metadata[partition[part]][['sig_len']].copy()
    sample = initialize.sample_data(H, sig_len, part)
    _sig = sig_data[partition[part]].copy()
    _meta = metadata[partition[part]].copy()
    _dx = diagnoses[partition[part]].copy()
    data[part] = initialize.dataframes_to_tensors(H, sample, _sig, _meta, _dx)

path = '/scr1/mimic/initial_data_{}.pkl'.format(H['epochs'])
with open(path, 'wb') as f:
    pickle.dump(data, f)