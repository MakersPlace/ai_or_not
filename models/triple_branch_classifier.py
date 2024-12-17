import tensorflow as tf
from layers.fourier_transforms import FourierTransformLayer
from models.base_model import BaseModel
from models.efficientnet_v2 import EfficientNetV2B3
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2


class TripleBranchClassifier(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def get_backbone(self, weights):
        backbone = EfficientNetV2B3(
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
        dense_nodes = self.config["N_DENSE"] * 3
        x = GlobalAveragePooling2D()(input)
        x = Dense(dense_nodes, activation="relu", kernel_regularizer=L2(self.config["REGULARIZATION_RATE"]))(x)
        x = BatchNormalization()(x)
        x = Dense(len(self.config["CLASS_NAMES"]))(x)
        x = Activation("softmax", dtype="float32", name="predictions")(x)

        return x

    def build_classifier(self):
        # --------------------------------------------------------------
        # Input
        # --------------------------------------------------------------
        input_shape = self.get_input_shape()
        rgb_input = Input(shape=input_shape)

        # --------------------------------------------------------------
        # RGB Tower - Tower 1
        # --------------------------------------------------------------
        rgb_backbone = self.get_backbone(weights=self.config["WEIGHTS"])
        rgb_embedding = rgb_backbone(rgb_input)

        # --------------------------------------------------------------
        # RGB FFT Tower -  - Tower 2
        # --------------------------------------------------------------
        fft_backbone = self.get_backbone(weights=self.config["WEIGHTS"])
        fourier_transform = FourierTransformLayer()
        rgb_fft = fourier_transform(rgb_input)
        rgb_fft_embedding = fft_backbone(rgb_fft)

        # --------------------------------------------------------------
        # Noise Residual Tower - Tower 3
        # --------------------------------------------------------------
        noise_backbone = self.get_backbone(weights=self.config["WEIGHTS"])
        fourier_transform = FourierTransformLayer()
        denoiser = self.get_denoiser()
        x = denoiser(rgb_input)
        noise_fft = fourier_transform(x)
        noise_fft_embedding = noise_backbone(noise_fft)

        # --------------------------------------------------------------
        # concatenate embeddings
        # --------------------------------------------------------------
        combined_embedding = tf.keras.layers.concatenate([rgb_embedding, rgb_fft_embedding, noise_fft_embedding])

        # --------------------------------------------------------------
        # Fully Connected Layers
        # --------------------------------------------------------------
        output = self.get_fc_layers(combined_embedding)

        model = tf.keras.Model(
            inputs=[rgb_input],
            outputs=[output],
            name="triple_branch_classifier",
        )

        # Model Summary
        self.print_model_summary(model)

        return model

    def train(self, train_dataset, validation_dataset):
        train_dataset = self.prepare_dataset(train_dataset)
        validation_dataset = self.prepare_dataset(validation_dataset)
        self.model = self.build_classifier()

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
