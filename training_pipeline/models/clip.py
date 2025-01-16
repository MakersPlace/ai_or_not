import tensorflow as tf
from keras import Input
from keras import Model
from keras.initializers import GlorotUniform
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Adam
from keras.regularizers import L2
from models.base_model import BaseModel
from transformers import TFCLIPVisionModel

MODEL_NAME = "openai/clip-vit-base-patch32"
CACHE_DIR = "/tmp"


class TFCLIPModelLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TFCLIPModelLayer, self).__init__(**kwargs)
        self.model = TFCLIPVisionModel.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
        )
        self.model.trainable = False

    def call(self, inputs):
        transposed_inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        outputs = self.model(transposed_inputs)
        return outputs.pooler_output

    def compute_output_shape(self, input_shape):
        return [None, 768]


# extend callback factory with a new method
class CLIPClassifier(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def get_backbone(self, weights):
        return TFCLIPModelLayer()

    def build_classifier(self):
        # --------------------------------------------------------------
        # Input layers
        # --------------------------------------------------------------
        input_shape = self.get_input_shape()
        input_ids = Input(
            shape=input_shape,
            dtype=tf.float32,
            name="input_ids",
        )

        # --------------------------------------------------------------
        # Backbone Layers
        # --------------------------------------------------------------
        pooled_output = TFCLIPModelLayer()(input_ids)

        # --------------------------------------------------------------
        # Head initialization
        # --------------------------------------------------------------
        dense_layer = Dense(
            units=len(self.config["CLASS_NAMES"]),
            activation="softmax",
            kernel_regularizer=L2(self.config["REGULARIZATION_RATE"]),
            kernel_initializer=GlorotUniform(seed=self.config["SEED"]),
        )(pooled_output)
        model = Model(
            inputs=input_ids,
            outputs=dense_layer,
            name="clip_classifier",
        )

        self.print_model_summary(model)

        return model

    def train(self, train_dataset, validation_dataset):
        train_dataset = self.prepare_dataset(train_dataset)
        validation_dataset = self.prepare_dataset(validation_dataset)

        self.model = self.build_classifier()

        # load best weights if available
        self.load_pretrained_weights()

        self.model.compile(
            optimizer=Adam(learning_rate=self.config["LEARNING_RATE"]),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()],
        )

        callbacks = self.get_callbacks(self.model.name)

        self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config["N_EPOCHS"],
            callbacks=callbacks,
            verbose=self.config["VERBOSE"],
        )

        return self.save_and_load_model(self.model)
