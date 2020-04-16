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

parts = ['train', 'validation']
tensors, metadata, priors = initialize.run(H, parts=parts)
dataset = {part: data_pipeline.build(H, tensors[part], part) for part in parts}
model = conv_model.build(H, priors)
model.summary()

log_dir = '/scr-ssd/tflogs/' + model_id
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)

checkpoint_args = {
    'filepath': '/scr1/checkpoints/' + model_id + '_{epoch:05d}.ckpt',
    'save_weights_only': True,
    'period': 2**7
}
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(**checkpoint_args)        
    
model.fit(
    dataset['train'],
    validation_data = dataset['validation'],
    epochs = 2**20,
    steps_per_epoch = 2**H['steps_per_epoch_log2'],
    validation_steps = 2**H['validation_steps_log2'],
    validation_freq = range(0, 2**20, 2**H['validation_frequency_log2']),
    callbacks = [tensorboard_callback, checkpoint_callback],
    verbose = 0,
)
