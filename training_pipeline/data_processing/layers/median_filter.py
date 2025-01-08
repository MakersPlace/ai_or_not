# Write a custom kers layer to blur the image
#
# The layer will take an image as input and return a blurred image as output

import tensorflow as tf
from tensorflow.keras import layers


class MedianFilter(layers.Layer):
    def __init__(self, blur_range, probability, **kwargs):
        super(MedianFilter, self).__init__(**kwargs)
        self.blur_range = blur_range
        self.probability = probability

    def call(self, input):
        # get the random frequency
        probability = tf.random.uniform(shape=[], minval=0, maxval=1)

        # if the frequency is greater than the set frequency, return the original image
        if probability > self.probability:
            return input

        # get the random blur
        blur = tf.random.uniform(shape=[], minval=self.blur_range[0], maxval=self.blur_range[1], dtype=tf.int32)

        # blur each image in the batch
        blurred_image = tf.image.median_filter(input, blur)

        # return blurred images
        return blurred_image

    def get_config(self):
        parent_config = super().get_config()
        return {
            **parent_config,
            "blur_range": self.blur_range,
            "frequency": self.probability,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
