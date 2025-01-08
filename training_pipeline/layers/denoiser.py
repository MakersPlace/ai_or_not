import tensorflow as tf


# Layer to to Extract Noise Residual from images
class Denoiser(tf.keras.layers.Layer):
    def __init__(self, denoiser_model_path, **kwargs):
        super(Denoiser, self).__init__(**kwargs)
        self.denoiser = tf.keras.models.load_model(denoiser_model_path, compile=False)
        self.denoiser.trainable = False

    def call(self, inputs):
        denoised_images = self.denoiser(inputs)
        residual_images = tf.subtract(inputs, denoised_images)
        normalized_image = tf.divide(tf.add(residual_images, 1), 2)

        return normalized_image
