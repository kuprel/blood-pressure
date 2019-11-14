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

# path = initialize.DATA_ROOT + 'initial_data_{}.hdf'.format(H['epochs'])
# data = initialize.load(H, path)
data = initialize.run(H)
data = initialize.dataframes_to_tensors(H, data)
data = data_pipeline.build(H, data)

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
    validation_freq = H['validation_frequency'],
    callbacks = [tb_callback]
)
