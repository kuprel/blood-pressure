import datetime
import pickle
import tensorflow as tf
import initialize
import data_pipeline
import conv_model
import sys

tf.debugging.set_log_device_placement(True)

time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_id = sys.argv[1] + '_' + time_str

H = initialize.load_hypes(model_id)

initial_data_path = '/scr1/mimic/initial_data/'
sig_data, metadata = initialize.load_initial_data(load_path=initial_data_path)

diagnosis = initialize.load_diagnosis(H['icd_codes'], metadata)
diagnosis = initialize.augment_diagnosis(diagnosis, metadata)
diagnosis = initialize.fix_diagnosis(diagnosis)
diagnosis = initialize.conform_diagnosis(diagnosis, metadata)

partition = initialize.load_partition(H, sig_data)

dataset = {}
for part in ['train', 'validation']:
    I = partition[part]
    row_lengths = initialize.get_row_lengths(metadata[I])
    args = [metadata[I], sig_data[I], diagnosis[I], row_lengths]
    tensors = initialize.get_tensors(H, *args)
    dataset[part] = data_pipeline.build(H, tensors, part)

model = conv_model.build(H, diagnosis_codes=list(diagnosis))
model.summary()

log_dir = '/scr-ssd/tflogs/' + model_id
tb_callback = tf.keras.callbacks.TensorBoard(log_dir)
        
model.fit(
    dataset['train'],
    validation_data = dataset['validation'],
    epochs = 2**20,
    steps_per_epoch = H['steps_per_epoch'],
    validation_steps = 1,
    validation_freq = 2**4,
    callbacks = [tb_callback],
    verbose = 0,
)
