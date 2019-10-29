import datetime
import pickle
import tensorflow as tf
import initialize
import data_pipeline
import conv_model
import sys

tf.debugging.set_log_device_placement(True)

model_id = sys.argv[1] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

H = initialize.load_hypes(model_id)

path = '/scr-ssd/mimic/initial_data_{}.pkl'.format(H['epochs'])
with open(path, 'rb') as f:
    data = pickle.load(f)

for k in ['train', 'validation']:
    data[k] = data_pipeline.build(H, data[k], k)
    
model = conv_model.build(H)
model.summary()

log_dir = '/scr-ssd/tflogs/' + model_id
tb_callback = tf.keras.callbacks.TensorBoard(log_dir)
        
model.fit(
    data['train'],
    validation_data = data['validation'],
    epochs = H['epochs'],
    steps_per_epoch = H['steps_per_epoch'],
    validation_steps = H['validation_steps'],
    callbacks = [tb_callback]
)
