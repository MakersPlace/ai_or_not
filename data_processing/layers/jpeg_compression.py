from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers


class JPEGCompression(layers.Layer):
    def __init__(self, quality_range, probability, seed, **kwargs):
        super(JPEGCompression, self).__init__(dtype=tf.float32, **kwargs)
        self.quality_range = quality_range
        self.probability = probability
        self.seed = seed

    # write a function to compress the image using PIL library to given quality
    def compress_image(self, image, quality):
        image = image.numpy()
        image = Image.fromarray(image)
        image_bytes = BytesIO()
        image.save(image_bytes, format="jpeg", quality=quality)
        image = Image.open(image_bytes)
        np_array = np.asarray(image)
        return np_array

    def call(self, input):
        probability = tf.random.uniform(shape=[], minval=0, maxval=1, seed=self.seed)

        if probability > self.probability:
            return input

        quality = tf.random.uniform(
            shape=[], minval=self.quality_range[0], maxval=self.quality_range[1], seed=self.seed, dtype=tf.int32
        )

        compressed_image = tf.image.adjust_jpeg_quality(input, quality)

        return compressed_image

    def get_config(self):
        parent_config = super().get_config().copy()
        return {
            **parent_config,
            "quality_range": self.quality_range,
            "frequency": self.probability,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
