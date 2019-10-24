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

with open('/scr-ssd/mimic/initial_data.pkl', 'rb') as f:
    data = pickle.load(f)

for k in ['train', 'validation']:
    data[k] = data_pipeline.build(H, data[k], k)
    
model = conv_model.build(H)
model.summary()

tb_callback = tf.keras.callbacks.TensorBoard('/scr-ssd/tflogs/' + model_id)

model.fit(
    data['train'],
    validation_data = data['validation'],
    epochs = H['epochs'],
    steps_per_epoch = H['steps_per_epoch'],
    validation_steps = H['validation_steps'],
    callbacks = [tb_callback]
)
