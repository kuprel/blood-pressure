import datetime
import tensorflow as tf

import prepare_data
import initialize
import data_pipeline
import conv_model

tf.debugging.set_log_device_placement(True)

model_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

H = initialize.load_hypes(model_id)
metadata = initialize.load_metadata(H)
sig_data = initialize.load_sig_data(H, metadata)
partition = initialize.load_partition(H, metadata)

initialize.describe_data_size(metadata)

dataframes = {
    'train': initialize.sample_data(H, sig_data[partition['train']]),
    'validation': initialize.sample_data(H, sig_data[partition['validation']], is_validation=True)
}

tensors = {
    'train': initialize.dataframe_to_tensors(H, dataframes['train']),
    'validation': initialize.dataframe_to_tensors(H, dataframes['validation']),
}

dataset = {
    'train': data_pipeline.build(H, tensors['train']),
    'validation': data_pipeline.build(H, tensors['validation'], is_validation=True),
}

model = conv_model.build(H)
model.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard('/scr-ssd/tflogs/' + model_id)

model.fit(
    dataset['train'],
    validation_data = dataset['validation'],
    epochs = H['epochs'],
    steps_per_epoch = H['steps_per_epoch'],
    validation_steps = H['validation_steps'],
    callbacks = [tensorboard_callback]
)