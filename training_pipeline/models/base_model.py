import logging as log
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.integration.keras import WandbMetricsLogger


class BaseModel:
    PRETRAINED_WEIGHTS_PATH = None

    def __init__(self, config):
        self.config = config
        self.saved_model_filepath = Path(self.config["MODEL_DIR"]) / "saved_model"
        self.best_weights_filepath = Path(self.config["CHECKPOINTS_PATH"]) / "best_weights"
        self.checkpoints_filepath = Path(self.config["CHECKPOINTS_PATH"]) / "ckpt-{epoch:02d}-{val_loss:.3f}"
        self.monitor = self.config["TRAINING_MONITOR"]

        log.info(f"\tSaved model filepath: {self.saved_model_filepath}")
        log.info(f"\tBest weights filepath: {self.best_weights_filepath}")
        log.info(f"\tCheckpoints filepath: {self.checkpoints_filepath}")

    def build_classifier(self):
        raise NotImplementedError

    def prepare_dataset(self, dataset):
        dataset = dataset.batch(self.config["BATCH_SIZE"])
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_backbone(self, weights):
        log.info(f"\tLoading backbone with weights: {weights}")
        backbone = tf.keras.applications.EfficientNetV2S(
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

    def get_input_shape(self):
        return (
            self.config["IM_SIZE"],
            self.config["IM_SIZE"],
            self.config["CHANNELS"],
        )

    def get_model_filepath(self, file_path, model_name=""):
        if model_name:
            return f"{file_path}_{model_name}"
        else:
            return file_path

    def get_weights_path(self, file_path, model_name):
        return f"{self.get_model_filepath(file_path, model_name=model_name)}.weights.h5"

    def get_callbacks(self, model_name=""):
        # Reduce LR On no Improvement
        reduce_lr = ReduceLROnPlateau(
            monitor=self.monitor,
            factor=0.1,
            patience=self.config["PATIENCE"],
            verbose=self.config["VERBOSE"],
            mode="min",
            min_delta=self.config["MIN_DELTA"],
            cooldown=0,
            min_lr=1e-15,
        )

        # Early Stopping
        early_stopping = EarlyStopping(
            monitor=self.monitor,
            min_delta=self.config["MIN_DELTA"],
            patience=(self.config["PATIENCE"] * 2),
            verbose=self.config["VERBOSE"],
            mode="min",
            baseline=None,
            restore_best_weights=True,
        )

        best_weights_checkpoint = ModelCheckpoint(
            self.get_weights_path(
                self.best_weights_filepath,
                model_name=model_name,
            ),
            monitor=self.monitor,
            verbose=self.config["VERBOSE"],
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
        )

        model_checkpoint = ModelCheckpoint(
            self.get_weights_path(
                self.checkpoints_filepath,
                model_name=model_name,
            ),
            monitor=self.monitor,
            verbose=self.config["VERBOSE"],
            save_best_only=False,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
        )

        return [
            reduce_lr,
            early_stopping,
            best_weights_checkpoint,
            model_checkpoint,
            WandbMetricsLogger(),
        ]

    def save_and_load_model(self, model):
        log.info("-------------------------------------------------")
        best_weights_filepath = self.get_weights_path(
            self.best_weights_filepath,
            model_name=model.name,
        )
        model.load_weights(best_weights_filepath)
        log.info(f"Loaded best weights from {best_weights_filepath}")

        saved_model_filepath = self.get_model_filepath(self.saved_model_filepath, model_name=model.name)
        saved_model_filepath = f"{saved_model_filepath}.keras"
        model.save(saved_model_filepath)
        log.info(f"Saved model in SavedModel format {saved_model_filepath}")

        model = tf.keras.saving.load_model(saved_model_filepath)
        log.info(f"Loaded SavedModel model from {saved_model_filepath}")

        return model

    def load_pretrained_weights(self):
        result = False
        # load best weights and adjust parameters for fine-tuning
        log.info("-------------------------------------------------")
        if (
            self.config["LOAD_PRETRAINED_WEIGHTS"]
            and self.PRETRAINED_WEIGHTS_PATH
            and self.config["IM_SIZE"] == 224
            and Path(self.PRETRAINED_WEIGHTS_PATH).exists()
        ):
            file_path = Path(self.PRETRAINED_WEIGHTS_PATH) / f"best_weights_{self.model.name}"
            log.info(f"Loading pretrained weights from {file_path}")
            self.model.load_weights(file_path)
            self.config["LEARNING_RATE"] = self.config["LEARNING_RATE"] / 100
            log.info(f"Adjusted learning rate to {self.config['LEARNING_RATE']}")
            result = True
        else:
            log.info(
                f"Could not find pretrained weights for ImageSize[{self.config['IM_SIZE']}]"
                + f" in {self.PRETRAINED_WEIGHTS_PATH}"
            )
        log.info("=================================================")

        return result

    def print_model_summary(self, model):
        # Model Summary
        log.info(f"\t{model.summary(show_trainable=True, line_length=150)}")
        log.info("\t**************** Model Layers ****************")
        log.info("\t------------------------------------------------------------------")
        log.info("\tLayer Name".ljust(40) + "Compute".ljust(10) + "Variable".ljust(10))
        log.info("\t==================================================================")
        for layer in model.layers:
            log.info(
                "\t"
                + layer.name.ljust(40)
                + str(layer.dtype_policy.compute_dtype).ljust(10)
                + str(layer.dtype_policy.variable_dtype).ljust(10)
            )

        log.info("\t------------------------------------------------------------------")
        log.info("\t**************** Model Dtypes ****************")
        log.info("\t==================================================================")
        log.info("\t   ".ljust(40) + "Compute".ljust(10) + "Variable".ljust(10))
        log.info(
            f"\tDefault Dtype:".ljust(40)
            + f"{str(model.dtype_policy.compute_dtype)}".ljust(10)
            + f"{str(model.dtype_policy.variable_dtype)}".ljust(10)
        )
        log.info("\n")
