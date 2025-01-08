# Main training script for the AI or Not classifier
import argparse
import logging as log
import os
from datetime import datetime

import tensorflow as tf
from config import ARCHITECTURES
from config import DEFAULT_PIPELINE_NAME
from config import ModelArchitectureEnum
from config import get_config
from config import get_path_config
from config import setup_wandb
from data_processing.data_loader import DataLoader
from data_processing.evaluate_model import evaluate_model

# WANDB Constants
JOB_TYPE_SUFFIX = "rgb"
RUNTIME_DATE_SUFFIX = "%m%d_%H%M"
RUN_NAME_SUFFIX = datetime.now().strftime(RUNTIME_DATE_SUFFIX)
CONFIGURATION = get_config()
PATH_CONFIG = get_path_config(DEFAULT_PIPELINE_NAME)


def train_model(train_dataset, validation_dataset, current_config):
    """Generate a simple model"""
    model_architecture = current_config["MODEL_ARCHITECTURE"]
    if not (
        model_architecture >= ModelArchitectureEnum.VISIBLE_FEATURES
        and model_architecture <= ModelArchitectureEnum.NOISE_RESIDUAL
    ):
        raise ValueError(f"Invalid model architecture: {model_architecture}")

    multi_label_classifier_model = ARCHITECTURES[model_architecture](current_config)
    trained_model = multi_label_classifier_model.train(train_dataset, validation_dataset)

    return trained_model


def load_training_data(current_config):
    data_loader = DataLoader(current_config)
    train_dataset = data_loader.load_and_augment_dataset(current_config["TF_TRAIN_DATASET_PATH"])
    validation_dataset = data_loader.load_dataset(current_config["TF_VALIDATION_DATASET_PATH"])
    return train_dataset, validation_dataset


def update_path_config_with_pipeline_name(pipeline_name):
    if pipeline_name != DEFAULT_PIPELINE_NAME:
        log.info(f"Pipeline Name: {args.pipeline_name}")
        # Reinstantiate the configuration with the new pipeline name
        global PATH_CONFIG
        PATH_CONFIG = get_path_config(args.pipeline_name)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--pipeline_name", help="Name of the pipeline run", default=DEFAULT_PIPELINE_NAME)
    parser.add_argument("--environment", help="Decides which AWS profile to use", default="local")

    return parser.parse_known_args()


def get_directory_config(args):
    # ----------------- S3 Directories ----------------- #
    train_data_prefix = args.train
    train_dataset_path = f"{train_data_prefix}/training_dataset"
    validation_dataset_path = f"{train_data_prefix}/validation_dataset"
    test_dataset_path = f"{train_data_prefix}/testing_datasets"
    denoiser_model_path = "./cache/dncnn_s1_1028_0147_ai_48/saved_model"

    return {
        "TF_TRAIN_DATASET_PATH": train_dataset_path,
        "TF_VALIDATION_DATASET_PATH": validation_dataset_path,
        "TF_TEST_DATASET_PATH": test_dataset_path,
        "DENOISER_MODEL_PATH": denoiser_model_path,
        "MODEL_DIR": args.sm_model_dir,
        "CHECKPOINTS_PATH": PATH_CONFIG["SM_CHECKPOINTS_PATH"],
    }


def setup_env_confg(args):
    # Set Global seed
    # Sets seeds for Python, Tensorflow and Numpy
    tf.keras.utils.set_random_seed(CONFIGURATION["SEED"])

    # --------- Directory Config ---------
    directory_config = get_directory_config(args)

    # --------- Set Mixed Precision ---------
    if CONFIGURATION["MIXED_PRECISION"]:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # ------------- Current Config -------------
    current_config = {**CONFIGURATION, **directory_config}

    return current_config


if __name__ == "__main__":
    args, unknown = parse_args()

    log.info(f"Pipeline Name: {args.pipeline_name} ")
    log.info(f"Environment: {args.environment} ")

    # Setup Env Config
    update_path_config_with_pipeline_name(args.pipeline_name)
    current_config = setup_env_confg(args)

    # Start Wandb Run
    wandb_run = setup_wandb(
        args.pipeline_name,
        args.environment,
        current_config,
        "train",
    )
    wandb_run.config.update(current_config)
    log.info("----------------- Training Config -----------------")
    log.info(f"Model Architecture: {current_config['MODEL_ARCHITECTURE']}")
    log.info(f"Image Size: {current_config['IM_SIZE']}")
    log.info(f"Batch Size: {current_config['BATCH_SIZE']}")
    log.info(f"Max Runtime: {int(current_config['MaxRuntimeInSeconds']/(60*60*24))} days")
    log.info(f"All Config: {current_config}")

    # Load training data
    train_dataset, validation_dataset = load_training_data(current_config)
    log.info("Datasets created successfully")

    # Train the model
    aiornot_classifier_model = train_model(train_dataset, validation_dataset, current_config)
    # aiornot_classifier_model = tf.keras.models.load_model(
    #     "/Users/skoneru/Downloads/400k_model/saved_model_rgb_classifier"
    # )

    # Test the model
    evaluate_model(aiornot_classifier_model, current_config, wandb_run)

    # Finish wandb run
    wandb_run.finish()

    # Completed trainning
    log.info("Completed training")
