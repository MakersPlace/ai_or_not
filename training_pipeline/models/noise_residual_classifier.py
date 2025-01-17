import tensorflow as tf
from layers.denoiser import Denoiser
from models.base_model import BaseModel
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2


class NoiseResidualClassifier(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build_classifier(self):
        # --------------------------------------------------------------
        # Input layers
        # --------------------------------------------------------------
        input_shape = self.get_input_shape()
        input_layers = [
            InputLayer(input_shape=input_shape),
            Rescaling(scale=2.0, offset=-1.0, dtype=tf.float32, name="rescaling"),
            Denoiser(self.config["DENOISER_MODEL_PATH"], dtype=tf.float32),
        ]

        # --------------------------------------------------------------
        # Backbone Layers
        backbone = [
            self.get_backbone(weights=self.config["WEIGHTS"]),
            GlobalAveragePooling2D(),
        ]

        # --------------------------------------------------------------
        # Head initialization
        # --------------------------------------------------------------
        head_layers = [
            Dropout(self.config["DROPOUT_RATE"], name="top_dropout"),
            Dense(
                len(self.config["CLASS_NAMES"]),
                kernel_regularizer=L2(self.config["REGULARIZATION_RATE"]),
                kernel_initializer=GlorotUniform(seed=self.config["SEED"]),
            ),
            Activation("softmax", dtype=tf.float32, name="predictions"),
        ]
        model = tf.keras.Sequential(input_layers + backbone + head_layers, name="noise_residual_classifier")

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
