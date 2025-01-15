import tensorflow as tf
from keras.applications import ConvNeXtTiny
from keras.initializers import GlorotUniform
from keras.layers import Dense
from keras.layers import InputLayer
from keras.layers import Rescaling
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Adam
from keras.regularizers import L2
from models.base_model import BaseModel


# extend callback factory with a new method
class VisibleConvNextClassifier(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def get_backbone(self, weights):
        backbone = ConvNeXtTiny(
            include_top=True,
            include_preprocessing=True,
            weights=weights,
            pooling="max",
            input_shape=(
                self.config["IM_SIZE"],
                self.config["IM_SIZE"],
                self.config["CHANNELS"],
            ),
            classes=len(self.config["CLASS_NAMES"]),
        )
        backbone.trainable = self.config["TRAIN_BACKBONE"]

        return backbone

    def build_classifier(self):
        # --------------------------------------------------------------
        # Input layers
        # --------------------------------------------------------------
        input_shape = self.get_input_shape()
        input_layers = [
            InputLayer(input_shape=input_shape),
            Rescaling(scale=2.0, offset=-1.0, dtype=tf.float32, name="rescaling"),
        ]

        # --------------------------------------------------------------
        # Backbone Layers
        backbone = [
            self.get_backbone(weights=self.config["WEIGHTS"]),
        ]

        # --------------------------------------------------------------
        # Head initialization
        # --------------------------------------------------------------
        head_layers = [
            # Dropout(self.config["DROPOUT_RATE"], name="top_dropout"),
            Dense(
                units=len(self.config["CLASS_NAMES"]),
                activation="softmax",
                kernel_regularizer=L2(self.config["REGULARIZATION_RATE"]),
                kernel_initializer=GlorotUniform(seed=self.config["SEED"]),
            ),
        ]
        model = tf.keras.Sequential(
            input_layers + backbone + head_layers,
            name="rgb_convnext_classifier",
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
