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
data = initialize.load_data(H, from_disk=True)

for part in ['train', 'validation']:
    data[part] = data_pipeline.build(H, data[part], part)
    
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
