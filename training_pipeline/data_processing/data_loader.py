import logging as log

import tensorflow as tf
import tensorflow_io as tfio  # importing for S3
from data_processing.layers.jpeg_compression import JPEGCompression


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.cache_filepath = str("/tmp/training_cache")

    def load_dataset(self, directory):
        # ----------------- Load Dataset ----------------- #
        current_dataset = tf.data.TFRecordDataset.load(
            path=str(directory),
            compression=self.config["COMPRESSION_TYPE"],
        )

        # print sample image with shape and dtype
        image = list(current_dataset.take(1))[0][0]
        log.info(f"Sample Image Size: {image.shape}")

        return current_dataset

    def print_sample(self, current_dataset):
        sample_element = list(current_dataset.take(1))[0][0]
        log.info(
            f"\tData Sample [ Shape:{sample_element.shape}   Dtype:{sample_element.dtype}   "
            + f"Min:{tf.math.reduce_min(sample_element)}   Max:{tf.math.reduce_max(sample_element)} ]"
        )

    def get_common_data_augmentation(self):
        return []

    def get_training_data_augmentation(self):
        data_augmentation = [
            # Trainging data has compressed images in LAION Dataset
            JPEGCompression(
                name="jpeg_compression",
                quality_range=(30, 100),
                probability=0.5,
                seed=self.config["SEED"],
            ),
            tf.keras.layers.Lambda(
                lambda image: tf.image.random_flip_left_right(
                    image,
                    seed=self.config["SEED"],
                ),
                name="random_flip_left_right",
                dtype=tf.float32,
            ),
        ]

        return data_augmentation

    def filter_non_empty_images(self, image, label):
        return tf.math.reduce_std(image) > 0.001

    def load_testing_dataset(self, test_datatset_directory):
        dataset = self.load_dataset(test_datatset_directory)

        dataset = dataset.filter(self.filter_non_empty_images)

        augmentation = self.get_common_data_augmentation()
        if len(augmentation) > 0:
            self.print_augmentation_details(augmentation)
            augmentation = tf.keras.Sequential(augmentation, name="data_augmentation")
            dataset = dataset.map(
                lambda image, label: (augmentation(image), label),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            log.info("Augmenting Test dataset")

        # ----------------- Optimize Dataset ----------------- #
        dataset = dataset.batch(self.config["BATCH_SIZE"])
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def load_and_augment_dataset(self, directory):
        # ----------------- Load Dataset ----------------- #
        dataset = self.load_dataset(directory)
        # ----------------- Filter Dataset ----------------- #
        dataset = dataset.filter(self.filter_non_empty_images)

        # ----------------- Dataset Augmentation ----------------- #
        common_augmentation = self.get_common_data_augmentation()
        training_augmentation = self.get_training_data_augmentation()
        augmentation = common_augmentation + training_augmentation

        if len(augmentation) > 0:
            self.print_augmentation_details(augmentation)
            augmentation = tf.keras.Sequential(augmentation, name="data_augmentation")
            dataset = dataset.map(
                lambda image, label: (augmentation(image), label), num_parallel_calls=tf.data.AUTOTUNE
            )

        return dataset

    def print_augmentation_details(self, layers):
        log.info("\t------------------------------------------------------------------")
        log.info("\t******************* Data Augmentation Layers *********************")
        log.info("\t------------------------------------------------------------------")
        log.info("\tLayer Name".ljust(40) + "Compute".ljust(10) + "Variable".ljust(10))
        log.info("\t==================================================================")
        for layer in layers:
            log.info(
                "\t"
                + layer.name.ljust(40)
                + str(layer.dtype_policy.compute_dtype).ljust(10)
                + str(layer.dtype_policy.variable_dtype).ljust(10)
            )
        log.info("\n")
