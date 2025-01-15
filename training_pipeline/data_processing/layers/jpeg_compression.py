import tensorflow as tf
from tensorflow.keras import layers


class JPEGCompression(layers.Layer):
    def __init__(self, quality_range, probability, seed, **kwargs):
        super(JPEGCompression, self).__init__(dtype=tf.float32, **kwargs)
        self.quality_range = quality_range
        self.probability = probability
        self.seed = seed

    def call(self, input):
        probability = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=1,
            seed=self.seed,
        )

        result = tf.math.less_equal(
            probability,
            tf.constant(self.probability),
        )

        return tf.cond(
            result,
            lambda: self.compress_image(input),
            lambda: input,
        )

    def compress_image(self, input):
        quality = tf.random.uniform(
            shape=[],
            minval=self.quality_range[0],
            maxval=self.quality_range[1],
            seed=self.seed,
            dtype=tf.int32,
        )

        return tf.image.adjust_jpeg_quality(input, quality)

    def compute_output_shape(self, input_shape):
        return input_shape

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
