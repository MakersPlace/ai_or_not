import tensorflow as tf
from layers.fourier_transforms import FourierTransformLayer
from models.base_model import BaseModel
from models.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2


class DualEnsembleClassifier(BaseModel):
    PRETRAINED_WEIGHTS_PATH = "./models/pre_trained/2_tower_model/weights"

    def __init__(self, config):
        super().__init__(config)

    def get_backbone(self, weights):
        backbone = EfficientNetV2S(
            include_top=False,
            weights=weights,
            input_tensor=None,
            include_preprocessing=False,
            input_shape=(
                self.config["IM_SIZE"],
                self.config["IM_SIZE"],
                self.config["CHANNELS"],
            ),
        )

        backbone.trainable = self.config["TRAIN_BACKBONE"]

        return backbone

    def get_fc_layers(self, input):
        x = Dense(len(self.config["CLASS_NAMES"]), kernel_regularizer=L2(self.config["REGULARIZATION_RATE"]))(input)
        x = Activation("softmax", dtype="float32", name="predictions")(x)

        return x

    def build_branch(self, fft_layer=None):
        # --------------------------------------------------------------
        # Input layers
        # --------------------------------------------------------------
        input_shape = self.get_input_shape()
        input_layers = [
            InputLayer(input_shape=input_shape),
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
            Dense(len(self.config["CLASS_NAMES"]), kernel_regularizer=L2(self.config["REGULARIZATION_RATE"])),
            Activation("softmax", dtype="float32", name="predictions"),
        ]

        if fft_layer:
            input_layers.append(fft_layer)
            name = "fft_classifier"
        else:
            name = "rgb_classifier"
        model = tf.keras.Sequential(input_layers + backbone + head_layers, name=name)

        self.print_model_summary(model)

        return model

    def train_model(self, model, train_dataset, validation_dataset):
        model.compile(
            optimizer=Adam(learning_rate=self.config["LEARNING_RATE"]),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()],
        )

        callbacks = self.get_callbacks(model.name)

        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config["N_EPOCHS"],
            callbacks=callbacks,
            verbose=self.config["VERBOSE"],
        )

        return self.save_and_load_model(model)

    def build_ensemble(self, rgb_branch, fft_branch):
        rgb_input = Input(shape=(self.config["IM_SIZE"], self.config["IM_SIZE"], self.config["CHANNELS"]))

        # Set Branches trainable to false
        rgb_branch.trainable = False
        fft_branch.trainable = False

        rgb_output = tf.keras.Sequential(rgb_branch.layers[0:2])(rgb_input)
        fft_output = tf.keras.Sequential(fft_branch.layers[0:3])(rgb_input)

        combined_layer = tf.keras.layers.concatenate([rgb_output, fft_output])
        outputs = self.get_fc_layers(combined_layer)

        model = tf.keras.Model(inputs=[rgb_input], outputs=outputs, name="ensemble_classifier")

        self.print_model_summary(model)

        return model

    def train(self, train_dataset, validation_dataset):
        train_dataset = self.prepare_dataset(train_dataset)
        validation_dataset = self.prepare_dataset(validation_dataset)

        rgb_branch = self.build_branch()
        trained_rgb_branch = self.train_model(rgb_branch, train_dataset, validation_dataset)

        fft_branch = self.build_branch(fft_layer=FourierTransformLayer())
        trained_fft_branch = self.train_model(fft_branch, train_dataset, validation_dataset)

        ensemble = self.build_enseble(trained_rgb_branch, trained_fft_branch)
        trained_ensemble = self.train_model(ensemble, train_dataset, validation_dataset)

        return trained_ensemble
