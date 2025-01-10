import argparse
import logging as log
import os
from datetime import datetime
from io import BytesIO
from typing import Any
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio  # importing for S3 support
from config import DEFAULT_PIPELINE_NAME
from config import TEST_DATASET_DRECTORIES
from config import TRAIN_DIRECTORIES
from config import get_config
from config import get_path_config
from config import setup_wandb
from PIL import Image

CONFIGURATION: Dict[str, Any] = get_config()
PATH_CONFIG: Dict[str, Any] = get_path_config(DEFAULT_PIPELINE_NAME)

IS_SAGEMAKER = CONFIGURATION["IS_SAGEMAKER"]
TF_SUPPORTED_FORMATS = set(CONFIGURATION["SUPPORTED_FORMATS"][:3])
PILLOW_SUPPORTED_FORMATS = set(CONFIGURATION["SUPPORTED_FORMATS"][3:])


# crop type enum
class CropType:
    RESIZE = 0
    CENTER_CROP = 1
    RANDOM_CROP = 2


def preprocess_image(file_path, crop_type):
    # Read the image file
    image_data = tf.io.read_file(file_path)
    im_size = CONFIGURATION["IM_SIZE"]

    # Decode the image using PIL
    def decode_image(image_data):
        try:
            image = Image.open(BytesIO(image_data.numpy()))

            if image.format.lower() in TF_SUPPORTED_FORMATS:
                image = tf.image.decode_image(
                    image_data, channels=CONFIGURATION["CHANNELS"], dtype=tf.float32, expand_animations=False
                )
            elif image.format.lower() in PILLOW_SUPPORTED_FORMATS:
                # Convert to RGB
                if image.mode != "RGB":
                    image = image.convert("RGB")
                # normalize to 0-1
                image = np.array(image) / 255.0
            else:
                raise ValueError(f"Image format {image.format} not supported")
        except Exception as e:
            log.warning(f"Error decoding image: {file_path} - {e}")
            # crete dummy image
            image = tf.zeros(shape=[1, 1, CONFIGURATION["CHANNELS"]])

        return image

    image = tf.py_function(decode_image, inp=[image_data], Tout=tf.float32, name="decode_image_py_func")
    # Resize the image for training
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    if width < im_size or height < im_size:
        image = tf.image.resize_with_pad(image, im_size, im_size)

    if crop_type == CropType.RESIZE:
        # Resize the image. Do not crop the image
        image = tf.image.resize_with_pad(image, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"])
    elif crop_type == CropType.CENTER_CROP:
        # Center crop the image for testing
        image = tf.image.resize_with_crop_or_pad(image, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"])
    elif crop_type == CropType.RANDOM_CROP:
        # Random crop for training
        image = tf.image.random_crop(
            value=image,
            size=[CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], CONFIGURATION["CHANNELS"]],
            seed=CONFIGURATION["SEED"],
        )
    else:
        raise ValueError(f"Invalid crop type: {crop_type}")

    return image


def visualize_dataset(samples, class_names, file_name=None):
    plt.figure(figsize=(12, 12))
    for index, (image, label) in enumerate(samples, start=1):
        plt.subplot(4, 4, index)
        plt.imshow(image)
        title = class_names[int(label)]
        plt.title(title)
        plt.axis("off")

    # Generate current EPOCH timestamp if file name is not provided
    if file_name is None:
        file_name = int(datetime.now().timestamp())

    # Save plot as an image
    plt.savefig(f"{PATH_CONFIG['CACHE_PATH']}/{file_name}.png")

    # close the plot
    plt.close()


def print_sample_image_data(image):
    log.info(
        f"Image Shape: {image.shape}  "
        f"Dtype: {image.dtype}  "
        f"Min: {tf.reduce_min(image)}  "
        f"Max: {tf.reduce_max(image)}"
    )


def load_and_visualize_dataset(dataset, class_names, crop_type):
    samples = [(preprocess_image(file_path, crop_type), label) for file_path, label in dataset]
    print_sample_image_data(samples[0][0])
    visualize_dataset(samples, class_names)


# TF Dataset creation function
def get_dataset(metadata_file, label, percentage):
    metadata_file_path = tf.strings.join([str(PATH_CONFIG["INPUT_DATA_PATH"]), metadata_file], separator="/")

    dataset = tf.data.experimental.CsvDataset(
        metadata_file_path,
        record_defaults=[tf.string],
        header=True,
        field_delim=",",
        select_cols=[0],
    )

    file_path_parts = tf.strings.split(metadata_file_path, sep="/")
    parent_path = tf.strings.reduce_join(file_path_parts[:-1], separator="/")

    dataset = dataset.map(
        lambda file_path: (tf.strings.join([parent_path, file_path], separator="/"), label, metadata_file),
    )

    # filter out based on percentage
    dataset = dataset.filter(lambda x, y, z: tf.random.uniform(shape=[], seed=CONFIGURATION["SEED"]) < percentage)

    return dataset


def preprocess_and_save(dataset, shards, dataset_path, crop_type=CONFIGURATION["CROP_TYPE"]):
    dataset = dataset.map(
        lambda file_path, label: (preprocess_image(file_path, crop_type), label),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    def custom_shard_func(ignore1, ignore2):
        # random number between 0 and shards
        return tf.random.uniform(shape=[], maxval=shards, dtype=tf.int64, seed=CONFIGURATION["SEED"])

    # filter out empty images if shape is not correct
    dataset = dataset.filter(lambda x, y: tf.shape(x)[0] == CONFIGURATION["IM_SIZE"])

    log.info(f"Saving Dataset to {dataset_path}")
    prefetched_dataset = dataset.prefetch(tf.data.AUTOTUNE)
    prefetched_dataset.save(
        path=str(dataset_path), shard_func=custom_shard_func, compression=CONFIGURATION["COMPRESSION_TYPE"]
    )


def create_test_datasets():
    test_dataset_counts: Dict[str, int] = {}
    test_label_counts: Dict[int, int] = {}
    dataset_length = 0

    # TODO: Suppress warinig in package tensorflow/core/lib/png/png_io.cc:84]
    # PNG warning: iCCP: known incorrect sRGB profile
    for dataset_name, test_metadata_files in TEST_DATASET_DRECTORIES.items():
        log.info(f"Processing {dataset_name} Dataset")

        current_dataset: tf.data.Dataset = None
        for test_directory_path, test_label, test_percentage in test_metadata_files:
            # For test run, and directory path does not contain "midjourney", update the test percentage
            if CONFIGURATION["TEST_RUN"] and "midjourney" not in test_directory_path:
                test_percentage = CONFIGURATION["TEST_DATASET_PERCENTAGE"]

            if isinstance(test_directory_path, list):
                metadata_file_path = []
                for d in test_directory_path:
                    metadata_file_path.append(d)
            else:
                metadata_file_path = test_directory_path

            sharded_dataset = get_dataset(metadata_file_path, test_label, test_percentage)

            current_dataset_length = len(list(sharded_dataset))
            test_dataset_counts[dataset_name] = test_dataset_counts.get(dataset_name, 0) + current_dataset_length
            dataset_length += current_dataset_length
            test_label_counts[test_label] = test_label_counts.get(test_label, 0) + current_dataset_length

            log.info(f"Processed Directory - {test_directory_path} Count: {current_dataset_length}")
            files_list_dataset = sharded_dataset.map(
                lambda file_path, label, metadata_file: (file_path, label),
            )

            if len(list(files_list_dataset.take(4))) == 4:
                load_and_visualize_dataset(
                    files_list_dataset.take(4), CONFIGURATION["CLASS_NAMES"], CONFIGURATION["CROP_TYPE"]
                )

            # merge datasets
            if current_dataset is None:
                current_dataset = files_list_dataset
            else:
                current_dataset = current_dataset.concatenate(files_list_dataset)

        # COmpute max shards based on the dataset size
        max_shards = max(int(test_dataset_counts.get(dataset_name, 0) / 2_000), 1)
        # Create directory for the test dataset
        test_dataset_path = f"{PATH_CONFIG['TESTING_DATASET_PATH']}/{dataset_name}"
        os.makedirs(test_dataset_path, exist_ok=True)

        preprocess_and_save(current_dataset, max_shards, test_dataset_path, crop_type=CONFIGURATION["CROP_TYPE"])

        log.info(f"{dataset_name}: Total Count: {test_dataset_counts.get(dataset_name, 0)}")

    log.info(f"Test Dataset Length: {dataset_length}")
    # sort label counts by label
    test_label_counts_dict = dict(sorted(test_label_counts.items()))
    log.info(f"Test Label Counts: {test_label_counts_dict}")
    # sort directory counts by directory
    directory_counts = dict(sorted(test_dataset_counts.items()))
    log.info(f"Test Directory Counts: {directory_counts}")


def download_sample_image(sample_dataset):
    # Download a sample image to verify
    for sample_image_path, sample_label, sample_metadata_file in sample_dataset:

        image_file = tf.io.read_file(sample_image_path)
        image = Image.open(BytesIO(image_file.numpy()))
        sample_image_write_path = f"{PATH_CONFIG['CACHE_PATH']}/s3_sample_image.png"
        log.info(f"Saving Sample Image to {sample_image_write_path}")
        image.save(sample_image_write_path)
        image = Image.open(sample_image_write_path)
        log.info(
            f"Sample Image Read Path: {sample_image_path}   "
            f"Sample Image Write Path: {sample_image_write_path}   "
            f"Label: {sample_label}   "
            f"Metadata File: {sample_metadata_file}"
        )
        log.info(f"Details Image Size: {image.size}  Mode: {image.mode} Format: {image.format}")


def create_stats(files_list_dataset):
    # --------------------------------------------------------------
    # Generate Stats for the Dataset size
    # --------------------------------------------------------------
    directory_counts: Dict[str, int] = {}
    label_counts: Dict[int, int] = {}
    dataset_length = 0

    for _, test_label, metadata_file_path in files_list_dataset:
        test_label = test_label.numpy()
        metadata_file_path = metadata_file_path.numpy().decode("utf-8")  # type: ignore
        label_counts[test_label] = label_counts.get(test_label, 0) + 1
        dataset_length += 1
        trimmed_path = metadata_file_path.replace(str(PATH_CONFIG["INPUT_DATA_PATH"]), "")
        parts = trimmed_path.split(sep="/")
        directory_name = parts[0] + "_" + parts[-2] + "_" + parts[-1]
        directory_counts[directory_name] = directory_counts.get(directory_name, 0) + 1

    log.info(f"Total Dataset Length: {dataset_length}")
    label_counts = dict(sorted(label_counts.items()))
    log.info(f"Label Counts: {label_counts}")
    directory_counts = dict(sorted(directory_counts.items()))
    log.info(f"Directory Counts: {directory_counts}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", help="Name of the pipeline run", default=DEFAULT_PIPELINE_NAME)
    parser.add_argument("--environment", help="Decides which AWS profile to use", default="local")
    return parser.parse_known_args()


def create_directories(PATH_CONFIG):
    # Creates directories in S3 or Local
    tf.io.gfile.makedirs(PATH_CONFIG["CACHE_PATH"])
    log.info(f"Cache Path: {PATH_CONFIG['CACHE_PATH']}")
    tf.io.gfile.makedirs(PATH_CONFIG["TRAINING_DATASET_PATH"])
    log.info(f"Training Dataset Path: {PATH_CONFIG['TRAINING_DATASET_PATH']}")
    tf.io.gfile.makedirs(PATH_CONFIG["VALIDATION_DATASET_PATH"])
    log.info(f"Validation Dataset Path: {PATH_CONFIG['VALIDATION_DATASET_PATH']}")
    tf.io.gfile.makedirs(PATH_CONFIG["TESTING_DATASET_PATH"])
    log.info(f"Testing Dataset Path: {PATH_CONFIG['TESTING_DATASET_PATH']}")


def check_if_metadata_files_exist():
    stop_execution = False
    metadata_file_names = [metadata_file for metadata_file, _, _ in TRAIN_DIRECTORIES]
    for metadata_file_path in metadata_file_names:
        metadata_file_path = f"{PATH_CONFIG['INPUT_DATA_PATH']}/{metadata_file_path}"
        if not tf.io.gfile.exists(metadata_file_path):
            stop_execution = True
            log.error(f"File not found: {metadata_file_path}")

    for _, test_metadata_files in TEST_DATASET_DRECTORIES.items():
        test_directory_paths = [test_directory_path for test_directory_path, _, _ in test_metadata_files]
        for test_directory_path in test_directory_paths:
            if not isinstance(test_directory_path, list):
                test_directory_path = [(test_directory_path, None, None)]
            for d, _, _ in test_directory_path:
                metadata_file_path = f"{PATH_CONFIG['INPUT_DATA_PATH']}/{d}"
                if not tf.io.gfile.exists(metadata_file_path):
                    stop_execution = True
                    log.error(f"File not found: {metadata_file_path}")

    if stop_execution:
        raise FileNotFoundError("Stopped execution due to missing files")


def update_path_config_with_pipeline_name(pipeline_name):
    if pipeline_name != DEFAULT_PIPELINE_NAME:
        log.info(f"Pipeline Name: {args.pipeline_name}")
        # Reinstantiate the configuration with the new pipeline name
        global PATH_CONFIG
        PATH_CONFIG = get_path_config(args.pipeline_name)


def get_shard_count():
    lower_shards = CONFIGURATION["TEST_RUN"] and CONFIGURATION["APPROX_DATASET_SIZE"] < 100_000

    return (
        200 if lower_shards else 1000,
        50 if lower_shards else 200,
    )


def execute(args):
    # Setup WANDB
    wandb_run = setup_wandb(
        args.pipeline_name,
        args.environment,
        CONFIGURATION,
        "data_generation",
    )

    wandb_run.config.update(CONFIGURATION)

    # Set Global seed
    # Sets seeds for Python, Tensorflow and Numpy
    tf.keras.utils.set_random_seed(CONFIGURATION["SEED"])

    # Update the Gloabal PATH_CONFIG with the pipeline name
    update_path_config_with_pipeline_name(args.pipeline_name)

    # Shard
    training_data_shards, validation_data_shards = get_shard_count()

    # Sanity check for metadata files
    check_if_metadata_files_exist()

    # Create directories (S3 or Local)
    create_directories(PATH_CONFIG)

    # Shuffle Dataset directories
    np.random.shuffle(TRAIN_DIRECTORIES)

    # Instantiate the dataset wtih metadata files
    metadata_files_tuple, labels_tuple, shards_tuple = zip(*TRAIN_DIRECTORIES)
    metadata_files: List[str] = list(metadata_files_tuple)
    labels: List[int] = list(labels_tuple)
    shards: List[float] = list(shards_tuple)

    log.info(f"Metadata Files: {len(metadata_files)}")

    metadata_files_dataset = tf.data.Dataset.from_tensor_slices((metadata_files, labels, shards))

    # Interleave the metadata files contents to create the dataset
    files_list_dataset = metadata_files_dataset.interleave(
        lambda metadata_file, label, shards: get_dataset(metadata_file, label, shards),
        cycle_length=len(TRAIN_DIRECTORIES),
        block_length=8,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # If its a test run, filter to smaller size
    if CONFIGURATION["TEST_RUN"]:
        files_list_dataset = files_list_dataset.take(CONFIGURATION["APPROX_DATASET_SIZE"])

    # Download a sample image to verify image paths and labels are correct
    download_sample_image(files_list_dataset.take(1))

    # Create Stats for the dataset size and label counts
    create_stats(files_list_dataset)

    # Map the dataset to only include file path and label
    files_list_dataset = files_list_dataset.map(
        lambda file_path, label, metadata_file: (file_path, label),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Shuffle image files dataset
    shuffled_dataset = files_list_dataset.shuffle(
        buffer_size=100_000, reshuffle_each_iteration=False, seed=CONFIGURATION["SEED"]
    )

    # Split dataset into Training and Validation Datasets
    validation_dataset_length = int(CONFIGURATION["APPROX_DATASET_SIZE"] * CONFIGURATION["VALIDATION_SPLIT"])
    log.info(
        f"Train Dataset length: {CONFIGURATION['APPROX_DATASET_SIZE']  - validation_dataset_length}    "
        f"Validation Dataset Length: {validation_dataset_length}"
    )
    validation_dataset = shuffled_dataset.take(validation_dataset_length)
    training_dataset = shuffled_dataset.skip(validation_dataset_length)

    # Saving Validation dataset
    preprocess_and_save(validation_dataset, validation_data_shards, PATH_CONFIG["VALIDATION_DATASET_PATH"])

    # Load and visualize the validation dataset from disk to verify
    log.info("Loading Validation dataset for from Disk")
    validation_dataset = tf.data.TFRecordDataset.load(
        PATH_CONFIG["VALIDATION_DATASET_PATH"], compression=CONFIGURATION["COMPRESSION_TYPE"]
    )
    print_sample_image_data(list(validation_dataset.take(1))[0][0])
    visualize_dataset(validation_dataset.take(16), CONFIGURATION["CLASS_NAMES"], "validation_dataset")

    # Saving Training dataset
    preprocess_and_save(training_dataset, training_data_shards, PATH_CONFIG["TRAINING_DATASET_PATH"])

    # Load and visualize the training dataset from disk to verify
    log.info("Loading Training dataset for from Disk")
    training_dataset = tf.data.TFRecordDataset.load(
        str(PATH_CONFIG["TRAINING_DATASET_PATH"]), compression=CONFIGURATION["COMPRESSION_TYPE"]
    )
    print_sample_image_data(list(training_dataset.take(1))[0][0])
    visualize_dataset(training_dataset.take(16), CONFIGURATION["CLASS_NAMES"], "training_dataset")

    # Create Test Datasets
    create_test_datasets()

    log.info("----------------------------------")
    log.info("--- Data Generation Completed ---")
    log.info("----------------------------------")

    # Finish wandb run.
    wandb_run.finish()

    return


if __name__ == "__main__":

    args, unknown = parse_args()
    log.info(f"TF: {tf.__version__} ")
    log.info(f"TFIO: {tfio.__version__} ")
    log.info(f"Pipeline Name: {args.pipeline_name} ")
    log.info(f"Environment: {args.environment} ")

    execute(args)

    log.info("Completed Data Generation Step")
