import functools
import tensorflow as tf
from tensorflow.signal import hann_window
from audio_to_multiple_pose_gan.tf_layers import ConvNormRelu, UpSampling1D


def tf_mel_spectograms(x_audio):
    stft = tf.signal.stft(
        x_audio,
        400,
        160,
        fft_length=512,
        window_fn=functools.partial(hann_window, periodic=True),
        pad_end=False,
        name=None
    )
    stft = tf.abs(stft)
    mel_spect_input = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=tf.shape(input=stft)[2],
        sample_rate=16000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        dtype=tf.float32,
        name=None
    )
    input_data = tf.tensordot(stft, mel_spect_input, 1)
    input_data = tf.math.log(input_data + 1e-6)
    input_data = tf.expand_dims(input_data, -1)
    return input_data


## Audio to Pose Discriminators ##

def D_patchgan(x_pose, n_downsampling=2, norm='batch', reuse=False, is_training=False, scope='discriminator'):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        ndf = 64
        model = tf.keras.layers.Conv1D(x_pose, filters=ndf, kernel_size=4, strides=2, padding='same',
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.compat.v1.zeros_initializer(), activation=None)
        model = tf.nn.leaky_relu(model, alpha=0.2)

        for n in range(1, n_downsampling):
            nf_mult = min(2**n, 8)
            model = ConvNormRelu(model, ndf * nf_mult, type='1d', downsample=True, norm=norm,
                    leaky=True, is_training=is_training)

        nf_mult = min(2**n_downsampling, 8)
        model = ConvNormRelu(model, ndf * nf_mult, type='1d', k=4, s=1,
                norm=norm, leaky=True)

        model = tf.keras.layers.Conv1D(model, filters=1, kernel_size=4, strides=1,
                                 padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                 bias_initializer=tf.compat.v1.zeros_initializer(), activation=None)
        print('discrimiator model output size', model, 'scope', scope, 'reuse', reuse)
        return model


def audio_to_pose(input_dict, reuse=False, is_training=False):
    x_audio = input_dict["audio"]
    input_data = tf_mel_spectograms(x_audio)

    with tf.compat.v1.variable_scope('generator', reuse=reuse):
        with tf.compat.v1.variable_scope('downsampling_block1'):
            conv = ConvNormRelu(input_data, 64, type='2d', is_training=is_training)
            first_block = ConvNormRelu(conv, 64, type='2d', is_training=is_training)
            first_block = tf.keras.layers.MaxPool1D(first_block, 2, 2)

        with tf.compat.v1.variable_scope('downsampling_block2'):
            second_block = ConvNormRelu(first_block, 128, type='2d', is_training=is_training)
            second_block = ConvNormRelu(second_block, 128, type='2d', is_training=is_training)
            second_block = tf.keras.layers.MaxPool2D(second_block, 2, 2)

        with tf.compat.v1.variable_scope('downsampling_block3'):
            third_block = ConvNormRelu(second_block, 256, type='2d', is_training=is_training)
            third_block = ConvNormRelu(third_block, 256, type='2d', is_training=is_training)
            third_block = tf.keras.layers.MaxPool2D(third_block, 2, 2)

        with tf.compat.v1.variable_scope('downsampling_block4'):
            fourth_block = ConvNormRelu(third_block, 256, type='2d', is_training=is_training)
            fourth_block = ConvNormRelu(fourth_block, 256, type='2d', is_training=is_training, k=(3, 8), s=1,
                                        padding='valid')

            fourth_block = tf.compat.v1.image.resize_bilinear(
                fourth_block,
                (input_dict["pose"].get_shape()[1].value, 1),
                align_corners=False,
                name=None
            )
            fifth_block = tf.squeeze(fourth_block, axis=2)

        with tf.compat.v1.variable_scope('downsampling_block5'):
            fifth_block = ConvNormRelu(fifth_block, 256, is_training=is_training)
            fifth_block = ConvNormRelu(fifth_block, 256, is_training=is_training)

            sixth_block = tf.keras.layers.MaxPool1D(fifth_block, 2, 2)
            sixth_block = ConvNormRelu(sixth_block, 256, is_training=is_training)

            seventh_block = tf.keras.layers.MaxPool1D(sixth_block, 2, 2)
            seventh_block = ConvNormRelu(seventh_block, 256, is_training=is_training)

            eight_block = tf.keras.layers.MaxPool1D(seventh_block, 2, 2)
            eight_block = ConvNormRelu(eight_block, 256, is_training=is_training)

            ninth_block = tf.keras.layers.MaxPool1D(eight_block, 2, 2)
            ninth_block = ConvNormRelu(ninth_block, 256, is_training=is_training)

            tenth_block = tf.keras.layers.MaxPool1D(ninth_block, 2, 2)
            tenth_block = ConvNormRelu(tenth_block, 256, is_training=is_training)

            ninth_block = UpSampling1D(tenth_block) + ninth_block
            ninth_block = ConvNormRelu(ninth_block, 256, is_training=is_training)

            eight_block = UpSampling1D(ninth_block) + eight_block
            eight_block = ConvNormRelu(eight_block, 256, is_training=is_training)

            seventh_block = UpSampling1D(eight_block) + seventh_block
            seventh_block = ConvNormRelu(seventh_block, 256, is_training=is_training)

            sixth_block = UpSampling1D(seventh_block) + sixth_block
            sixth_block = ConvNormRelu(sixth_block, 256, is_training=is_training)

            fifth_block = UpSampling1D(sixth_block) + fifth_block
            fifth_block = ConvNormRelu(fifth_block, 256, is_training=is_training)

        with tf.compat.v1.variable_scope('decoder'):
            fifth_block = ConvNormRelu(fifth_block, 256, is_training=is_training)
            fifth_block = ConvNormRelu(fifth_block, 256, is_training=is_training)
            fifth_block = ConvNormRelu(fifth_block, 256, is_training=is_training)
            fifth_block = ConvNormRelu(fifth_block, 256, is_training=is_training)

        with tf.compat.v1.variable_scope('logits'):
            conv = tf.keras.layers.Conv1D(fifth_block, filters=98, kernel_size=1, strides=1,
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                    bias_initializer=tf.compat.v1.zeros_initializer(), padding='same', activation=None)
        return conv


def audio_to_pose_gans(input_dict, reuse=False, is_training=False):
    with tf.compat.v1.variable_scope('generator', reuse=reuse):
        norm = input_dict['args'].norm
        x_audio = input_dict['audio']
        input_data = tf_mel_spectograms(x_audio)

        with tf.compat.v1.variable_scope('audio_encoder'):
            with tf.compat.v1.variable_scope('downsampling_block1'):
                conv = ConvNormRelu(input_data, 64, type='2d', leaky=True, downsample=False, norm=norm,
                                    is_training=is_training)
                first_block = ConvNormRelu(conv, 64, type='2d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

            with tf.compat.v1.variable_scope('downsampling_block2'):
                second_block = ConvNormRelu(first_block, 128, type='2d', leaky=True, downsample=False, norm=norm,
                                            is_training=is_training)
                second_block = ConvNormRelu(second_block, 128, type='2d', leaky=True, downsample=True, norm=norm,
                                            is_training=is_training)

            with tf.compat.v1.variable_scope('downsampling_block3'):
                third_block = ConvNormRelu(second_block, 256, type='2d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)
                third_block = ConvNormRelu(third_block, 256, type='2d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

            with tf.compat.v1.variable_scope('downsampling_block4'):
                fourth_block = ConvNormRelu(third_block, 256, type='2d', leaky=True, downsample=False, norm=norm,
                                            is_training=is_training)
                fourth_block = ConvNormRelu(fourth_block, 256, type='2d', leaky=True, downsample=False, norm=norm,
                                            is_training=is_training, k=(3, 8), s=1, padding='valid')

                fourth_block = tf.compat.v1.image.resize_bilinear(
                    fourth_block,
                    (input_dict["pose"].get_shape()[1].value, 1),
                    align_corners=False,
                    name=None
                )
                fifth_block = tf.squeeze(fourth_block, axis=2)

            with tf.compat.v1.variable_scope('downsampling_block5'):
                fifth_block = ConvNormRelu(fifth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)
                fifth_block = ConvNormRelu(fifth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)

                sixth_block = ConvNormRelu(fifth_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

                seventh_block = ConvNormRelu(sixth_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                             is_training=is_training)

                eight_block = ConvNormRelu(seventh_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

                ninth_block = ConvNormRelu(eight_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

                tenth_block = ConvNormRelu(ninth_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

                ninth_block = UpSampling1D(tenth_block) + ninth_block
                ninth_block = ConvNormRelu(ninth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)

                eight_block = UpSampling1D(ninth_block) + eight_block
                eight_block = ConvNormRelu(eight_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)

                seventh_block = UpSampling1D(eight_block) + seventh_block
                seventh_block = ConvNormRelu(seventh_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                             is_training=is_training)

                sixth_block = UpSampling1D(seventh_block) + sixth_block
                sixth_block = ConvNormRelu(sixth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)

                fifth_block = UpSampling1D(sixth_block) + fifth_block
                audio_encoding = ConvNormRelu(fifth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)


        with tf.compat.v1.variable_scope('decoder'):
            model = ConvNormRelu(audio_encoding, 256, type='1d', leaky=True, downsample=False, norm=norm, is_training=is_training)
            model = ConvNormRelu(model, 256, type='1d', leaky=True, downsample=False, norm=norm, is_training=is_training)
            model = ConvNormRelu(model, 256, type='1d', leaky=True, downsample=False, norm=norm, is_training=is_training)
            model = ConvNormRelu(model, 256, type='1d', leaky=True, downsample=False, norm=norm, is_training=is_training)

        with tf.compat.v1.variable_scope('logits'):
            model = tf.keras.layers.Conv1D(model, filters=98, kernel_size=1, strides=1,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(), bias_initializer=tf.compat.v1.zeros_initializer(),
                    padding='same', activation=None
                    )
    print('generator output size', model)
    return model


models = {

    #####################################
    #       Audio to Pose               #
    #####################################
    'audio_to_pose': audio_to_pose,
    'audio_to_pose_gans': audio_to_pose_gans,

    #####################################
    # Audio to Pose Discriminator   #
    #####################################
    'D_patchgan': D_patchgan,
}

def get_model(name):
    return models[name]
