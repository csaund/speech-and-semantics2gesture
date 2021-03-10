import tensorflow.keras as keras
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
import pandas as pd
from tqdm import tqdm
import random
from scipy.io import wavfile

# Set seed for experiment reproducibility
seed = 80121        # Happy birthday Kevin
tf.random.set_seed(seed)
np.random.seed(seed)
# Create a model that simply takes in audio and tries to guess the speaker.
# that's it for now.
# load wav file from audio_fn, gets speaker name, and tries to guess the speaker.
# should be really easy and get 100% right away.


## simple audio tutorial:
# https://www.tensorflow.org/tutorials/audio/simple_audio
# download data
data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.', cache_subdir='data')

# get list of possible data labels, essentially, by listing the dir names of datadir
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

# create filenames object which is a tensor of shape TensorShape([8000]) (num samples)
# with each element being <tf.Tensor: shape=(), dtype=string, numpy=b'data\\mini_speech_commands\\yes\\b36c27c2_nohash_0.wav'>
# filenames just points to wav file audio
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
random.shuffle(filenames)

# this sneakily converts fns to tensors >:(
# filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])


# Create batches
train_files = filenames[:6400]
val_files = filenames[6400: 6400 + 800]
test_files = filenames[-800:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))


# decodes audio into Tensor
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


# and create the label for that audio
# but this turns it into tensors.
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


# and create the label for that audio, but not tensors
def get_string_label(file_path):
    parts = file_path.split(os.path.sep)
    return parts[-2]


# get waveform and label of a particular fp from filepaths tensor
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def create_input_from_fns(fns):
    labels = []
    wavs = []
    for fn in fns:
        labels.append(get_string_label(fn))
        _, wav = wavfile.read(fn)
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        wavs.append(wav)
    return wavs, labels


train_files = filenames[:6400]
val_files = filenames[6400: 6400 + 800]
test_files = filenames[-800:]
# requires filenames to be list of strings!!
wavs, labs = create_input_from_fns(train_files)
valx, valy = create_input_from_fns(val_files)       ## haven't tried using this yes
# use this raw as input/output to model.fit now just to see what's happening.



# was trying to mess with dfs before
#train_df = create_df_from_files(train_files)
#val_df = create_df_from_files(val_files)
#test_df = create_df_from_files(test_files)


# from tutorial, makes TF DS, lame.
# AUTOTUNE = tf.data.AUTOTUNE
# files_ds = tf.data.Dataset.from_tensor_slices(train_files)
# waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)


# for waveform, label in waveform_ds.take(1):
#     label = label.numpy().decode('utf-8')         # this actually gets the label from that waveform_ds built before


#### Spectrogram, but comes back in tensor form
def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


specs = [get_spectrogram(w) for w in wavs]


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    X = np.arange(16000, step=height + 1)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


#fig, axes = plt.subplots(2, figsize=(12, 8))
#timescale = np.arange(waveform.shape[0])
#axes[0].plot(timescale, waveform.numpy())
#axes[0].set_title('Waveform')
#axes[0].set_xlim([0, 16000])
#plot_spectrogram(spectrogram.numpy(), axes[1])
#axes[1].set_title('Spectrogram')
#plt.show()


# this is type ParallelMapDataset
#spectrogram_ds = waveform_ds.map(
#    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    # output_ds = output_ds.map(
    #     get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    return output_ds


#train_ds = spectrogram_ds
#val_ds = preprocess_dataset(val_files)
#test_ds = preprocess_dataset(test_files)

# batch size etc
batch_size = 64
#train_ds = train_ds.batch(batch_size)
#val_ds = val_ds.batch(batch_size)
#train_ds = train_ds.cache().prefetch(AUTOTUNE)
#val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Creating the model
#for spectrogram, _ in spectrogram_ds.take(1):
#  input_shape = spectrogram.shape
#print('Input shape:', input_shape)

num_labels = len(commands)

norm_layer = preprocessing.Normalization()      # No idea what these do but they're obviously important
# norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

# My own doing
# I try both specs and wavs here to try out the raw wav or the spectrogram tensor, but they
# lead to different errors below.
input_shape = tf.squeeze(specs[0]).shape

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm_layer,
    layers.Conv1D(32, 3, activation='relu'),    # changed all these to 1D
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

# tried a variety of using the model defined above and below with various difficulties that I couldn't
# really map to either except that changing to 1D seemed to work above.


# model = get_simple_convlstm(input_shape)

model.summary()

# compile, aka define the optimizer and loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# end with
# return self._dims[key].value
# IndexError: list index out of range
# now fit the thing
EPOCHS = 10
history = model.fit(
    # train_ds,
    x=wavs, y=labs,     # currently these are raw wav and strings, but also tried converting to tensors, np arrays
    #validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)



# building a custom model to make the train shape happy.
class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def get_simple_convlstm(input_shape):
    seq = keras.Sequential(
        [
            keras.Input(
                shape=input_shape
            ),  # Variable-length sequence of 40x40x1 frames
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.Conv3D(
                filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
            ),
        ]
    )

    # seq.compile(loss="binary_crossentropy", optimizer="adadelta")
    return seq
