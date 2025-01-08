# write a custom keras layer to compute the auto correlation of an image

import tensorflow as tf
from tensorflow.keras.layers import Layer


class CovarianceLayer(Layer):
    def call(self, inputs):
        batch_size, height, width, channels = tf.shape(inputs)
        autocorr = []

        for i in range(batch_size):
            autocorr_per_batch = []

            for c in range(channels):
                image = inputs[i, :, :, c]
                mean_val = tf.reduce_mean(image)
                demeaned_image = image - mean_val

                autocorr_per_channel = []

                for delta_y in range(height):
                    for delta_x in range(width):
                        shifted_image = tf.roll(tf.roll(demeaned_image, shift=-delta_y, axis=0), shift=-delta_x, axis=1)
                        valid_area = demeaned_image * shifted_image
                        update_value = tf.reduce_mean(valid_area)
                        autocorr_per_channel.append(update_value)

                autocorr_per_batch.append(tf.reshape(tf.convert_to_tensor(autocorr_per_channel), (height, width)))

            autocorr.append(tf.stack(autocorr_per_batch, axis=-1))

        return tf.stack(autocorr)

    def compute_output_shape(self, input_shape):
        return input_shape
