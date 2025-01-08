import tensorflow as tf


class FourierTransformLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(dtype=tf.float32, **kwargs)

    def compute_fourier(self, channel):
        # -----------------------------------------------
        # Compute fourier transform of a single channel
        channel_complex64 = tf.cast(channel, tf.complex64)
        fft = tf.signal.fft2d(channel_complex64)
        fft = tf.signal.fftshift(fft, axes=(1, 2))
        amplitude = 20 * tf.math.log(tf.math.abs(fft) + 1)

        # -----------------------------------------------
        # Noramlize the fourier transformed image
        min_values = tf.math.reduce_min(amplitude, axis=(1, 2), keepdims=True)
        max_values = tf.math.reduce_max(amplitude, axis=(1, 2), keepdims=True)
        numerator = amplitude - min_values
        denominator = max_values - min_values
        f_normalized = tf.divide(numerator, denominator)

        return f_normalized

    def call(self, inputs):
        # inputs are of shape (batch_size, height, width, channels)
        # compute fourier transform of an RGB image
        # split image into channels
        red_channel = inputs[:, :, :, 0]
        green_channel = inputs[:, :, :, 1]
        blue_channel = inputs[:, :, :, 2]

        # compute fourier transform of each channel
        red_channel_fourier = self.compute_fourier(red_channel)
        green_channel_fourier = self.compute_fourier(green_channel)
        blue_channel_fourier = self.compute_fourier(blue_channel)

        # stack fourier transformed channels
        fourier_transformed_image = tf.stack(
            [red_channel_fourier, green_channel_fourier, blue_channel_fourier], axis=-1
        )

        return fourier_transformed_image

    # ensure input shape is (batch_size, height, width, channels)
    def compute_output_shape(self, input_shape):
        return input_shape
