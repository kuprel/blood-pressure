import initialize
import pickle

H = initialize.load_hypes()
data = initialize.load_metadata(H)
partition = initialize.load_partition(H, data)
initialize.describe_data_size(data)
data = initialize.load_sig_data(H, data)

initial_data = {}
for part in ['train', 'validation']:
    dataframe = data[partition[part]].copy()
    dataframe = initialize.sample_data(H, dataframe, part)
    initial_data[part] = initialize.dataframe_to_tensors(H, dataframe)

with open('/scr-ssd/mimic/initial_data.pkl', 'wb') as f:
    pickle.dump(initial_data, f)