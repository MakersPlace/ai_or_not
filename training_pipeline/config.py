import logging as log
import os
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List

from models.convnext import VisibleConvNextClassifier
from models.invisible_features_classifier import InvisibleFeaturesClassifier
from models.noise_residual_classifier import NoiseResidualClassifier
from models.visible_features_classifier import VisibleFeaturesClassifier

import wandb

# --------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------
SECONDS_IN_DAY = 86_400

# --------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------
log.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=log.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --------------------------------------------------------------
# Model Architectures
# Boththe enum and the list of architectures should be in sync
# --------------------------------------------------------------
class ModelArchitectureEnum:
    VISIBLE_FEATURES = 0
    INVISIBLE_FEATURES = 1
    VISIBLE_CONV_NEXT = 2
    NOISE_RESIDUAL = 3


ARCHITECTURES = [
    VisibleFeaturesClassifier,
    InvisibleFeaturesClassifier,
    VisibleConvNextClassifier,
    NoiseResidualClassifier,
]

# --------------------------------------------------------------
# PIPELINE CONFIGURATION
# --------------------------------------------------------------
PIPELINE_PREFIX = "ai-or-not-pipeline"
RUN_NAME_SUFFIX = datetime.now().strftime("%m%d%H%M%S")
DEFAULT_PIPELINE_NAME = f"{PIPELINE_PREFIX}-{RUN_NAME_SUFFIX}"
IS_SAGEMAKER = Path("/opt/ml").exists()


# S3 PATH INFO
S3_BUCKET = "mp-ml-data-dev"
S3_COMMON_PREFIX = "finder/ai_or_not"

# LOCAL PATH INFO
LOCAL_PATH_PREFIX = "/Volumes/FD/ai_or_not"

# --------------------------------------------------------------
# DATASETS Configuration
# --------------------------------------------------------------
TRAIN_DIRECTORIES = [
    # -----------------------------------------------------------------------------------
    # Label 0 - Real
    # -----------------------------------------------------------------------------------
    ("aiornot/image_files/train/REAL_metadata.csv", 0, 1.0),  # 512x512  8K
    ("DIRE/train/imagenet/real_metadata.csv", 0, 0.1),  # 40K
    ("DIRE/train/celebahq/real_metadata.csv", 0, 0.2),  # 28K
    ("DIRE/train/lsun_bedroom/real_metadata.csv", 0, 0.2),  # 40K
    ("cifake/train/REAL_metadata.csv", 0, 0.2),  # 32x32 50k
    ("gans/train_0_real_metadata.csv", 0, 0.2),  # 360K Pro GAN
    ("gans/progan_val_0_real_metadata.csv", 1, 1),  # 256x256 4K
    ("ai-artbench/train/real_metadata.csv", 0, 1),  # 50 K
    ("laion/6.5_20231203_200K/train_metadata.csv", 0, 1),  # 200K
    ("laion/6.25_20231203_500K/train_metadata.csv", 0, 0.5),  # 400K
    # -----------------------------------------------------------------------------------
    # Label 1 - GANS
    # -----------------------------------------------------------------------------------
    ("DIRE/train/lsun_bedroom/stylegan_metadata.csv", 1, 1),  # 256x256 40K
    ("gans/train_1_fake_metadata.csv", 1, 1),  # 256x256 360K Pro GAN
    ("gans/progan_val_1_fake_metadata.csv", 1, 1),  # 256x256 4K
    ("FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K_metadata.csv", 1, 1),  # 512x512 80K
    # -----------------------------------------------------------------------------------
    # Label 2 - Diffusion
    # -----------------------------------------------------------------------------------
    ("cifake/train/FAKE_metadata.csv", 2, 0.2),  # 32x32 50k Generated by SD 1.4
    ("DIRE/train/imagenet/adm_metadata.csv", 2, 0.2),
    ("DIRE/train/lsun_bedroom/adm_metadata.csv", 2, 0.2),
    ("DIRE/train/lsun_bedroom/iddpm_metadata.csv", 2, 0.2),
    ("DIRE/train/lsun_bedroom/pndm_metadata.csv", 2, 0.2),
    # SD V1_5
    ("ai-artbench/train_AI_metadata.csv", 2, 1),
    ("aiornot/image_files/train/FAKE_metadata.csv", 2, 1),  # 512x512 # 10K
    (
        "FakeImageDataset/ImageData/train/SDv15R-CC1M/SDv15R-dpmsolver-25-1M/SDv15R-CC1M_metadata.csv",
        2,
        0.15,
    ),  # 512x512 1M Generated by SD 1.5
    ("DIRE/train/celebahq/sdv2_metadata.csv", 2, 1),  # 768x768  40K  sdv2
    # MD V5
    (
        "FakeImageDataset/ImageData/val/Midjourneyv5-5K/Midjourneyv5-5K_train_metadata.csv",
        2,
        1,
    ),  # 4K - Split for validation needed
    # IF V1
    (
        "FakeImageDataset/ImageData/train/IFv1-CC1M/IFv1-dpmsolver++-50-1M/IF-CC1M_metadata.csv",
        2,
        0.15,
    ),  # 512x512 30K
]

TEST_DATASET_DRECTORIES: Dict[str, List[tuple[str, int, float]]] = {
    # --------------------------------------------------------------
    # Competition Datasets
    # --------------------------------------------------------------
    "CIFAKE": [("cifake/test/REAL_metadata.csv", 0, 1), ("cifake/test/FAKE_metadata.csv", 2, 1)],
    "ARTBENCH": [("ai-artbench/test/real_metadata.csv", 0, 1), ("ai-artbench/test_AI_metadata.csv", 2, 1)],
    # --------------------------------------------------------------
    # Research Datasets
    # --------------------------------------------------------------
    "GANS": [
        ("gans/test_0_real_metadata.csv", 0, 1),
        ("gans/test_1_fake_metadata.csv", 1, 1),
    ],
    "DIRE": [
        ("DIRE/test/imagenet/real_metadata.csv", 0, 1),
        ("DIRE/test/celebahq/real_metadata.csv", 0, 1),
        ("DIRE/test/imagenet/adm_metadata.csv", 2, 1),
        ("DIRE/test/celebahq/if_metadata.csv", 2, 1),
        ("DIRE/test/lsun_bedroom/vqdiffusion_metadata.csv", 2, 1),
        ("DIRE/test/celebahq/dalle2_metadata.csv", 2, 1),
        ("DIRE/test/lsun_bedroom/dalle2_metadata.csv", 2, 1),
        ("DIRE/test/imagenet/sdv1_metadata.csv", 2, 1),
        ("DIRE/test/lsun_bedroom/sdv1_new_metadata.csv", 2, 1),
        ("DIRE/test/celebahq/sdv2_metadata.csv", 2, 1),
        ("DIRE/test/lsun_bedroom/sdv2_metadata.csv", 2, 1),
        ("DIRE/test/lsun_bedroom/midjourney_metadata.csv", 2, 1),
    ],
    # --------------------------------------------------------------
    # Other Datasets
    # --------------------------------------------------------------
    "FAKEID": [
        ("FakeImageDataset/ImageData/val/stylegan3-60K/stylegan3-60K_metadata.csv", 1, 1),
        ("FakeImageDataset/ImageData/val/cogview2-22K/cogview2-22K_metadata.csv", 2, 1),
        ("FakeImageDataset/ImageData/val/IF-CC95K/IF-CC95K_metadata.csv", 2, 1),
        ("FakeImageDataset/ImageData/val/SDv15-CC30K/SDv15-CC30K_metadata.csv", 2, 1),
        ("FakeImageDataset/ImageData/val/SDv21-CC15K/SDv21-CC15K/SDv2-dpmsolver-25-10K_metadata.csv", 2, 1),
        ("FakeImageDataset/ImageData/val/Midjourneyv5-5K/Midjourneyv5-5K_test_metadata.csv", 2, 1),
    ],
    # --------------------------------------------------------------
    # Dalle3 Dataset
    # --------------------------------------------------------------
    "Dalle3": [
        ("mp_datasets/dalle3/2023_10_29_metadata.csv", 2, 1),
    ],
    # --------------------------------------------------------------
    # MidJourney Dataset
    # --------------------------------------------------------------
    "MidJourney": [
        ("mp_datasets/midjourney/2023_10_29_metadata.csv", 2, 1),
    ],
    # --------------------------------------------------------------
    # LAION Dataset
    # --------------------------------------------------------------
    "LAION": [
        ("laion/6.5_20231203_200K/test_metadata.csv", 0, 1),
        ("laion/6.25_20231203_500K/test_metadata.csv", 0, 1),
    ],
}


def override_test_config(current_config):
    current_config["TEST_RUN"] = False

    if current_config["TEST_RUN"]:
        current_config["APPROX_DATASET_SIZE"] = 900_000  # 10_000
        current_config["N_EPOCHS"] = 20  # 2
        current_config["TEST_DATASET_PERCENTAGE"] = 1.0  # 0.01

        if current_config["IS_SAGEMAKER"]:
            current_config["MIXED_PRECISION"] = True
        else:
            current_config["MIXED_PRECISION"] = False


_CONFIG = {
    # ----------------- Model config ----------------- #
    "MODEL_ARCHITECTURE": ModelArchitectureEnum.VISIBLE_FEATURES,
    "IM_SIZE": 224,
    "MIXED_PRECISION": True,  # FP16 training
    "IS_SAGEMAKER": IS_SAGEMAKER,
    # ----------------- Training config ----------------- #
    "TRAINING_MONITOR": "val_loss",
    "N_EPOCHS": 60,
    "REGULARIZATION_RATE": 0.01,
    "DROPOUT_RATE": 0.1,
    "N_DENSE": 1280,
    "LEARNING_RATE": 0.00001,
    "PATIENCE": 3,
    "MIN_DELTA": 0.0001,
    # Default weights settings
    "LOAD_PRETRAINED_WEIGHTS": False,
    "WEIGHTS": "imagenet",
    "TRAIN_BACKBONE": True,
    "SEED": 9,
    "VERBOSE": 1,
    # ----------------- Data config ----------------- #
    "SUPPORTED_FORMATS": ["bmp", "gif", "jpeg", "png", "webp"],
    "APPROX_DATASET_SIZE": 1_725_138,
    "VALIDATION_SPLIT": 0.2,
    "CLASS_NAMES": ["REAL", "GAN", "DIFFUSION"],
    "BINARY_CLASS_NAMES": ["REAL", "FAKE"],
    "CHANNELS": 3,
    "SHARDS": 1_000,
    "COMPRESSION_TYPE": "GZIP",
    "CACHE_DATASET": False,
    "CROP_TYPE": 0,
    # ----------------- Wandb config ----------------- #
    # WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT are read from environment
    # variables and passed to the Pipeline Steps
    # Reference: https://docs.wandb.ai/guides/track/environment-variables/
}


def get_batch_size(current_config):
    if current_config["IM_SIZE"] == 128:
        if current_config["MIXED_PRECISION"]:
            return 384
        else:
            return 128
    elif current_config["IM_SIZE"] == 256:
        if current_config["MIXED_PRECISION"]:
            return 96
        else:
            return 32
    elif current_config["IM_SIZE"] == 384:
        if current_config["MIXED_PRECISION"]:
            return 32
        else:
            return 16
    elif current_config["IM_SIZE"] == 224:
        if current_config["MIXED_PRECISION"] and current_config["MODEL_ARCHITECTURE"] < 5:
            return 128  # larger batches might reduce test accuracy
        else:
            return 32
    else:
        raise ValueError(f"Invalid image size: {current_config['IM_SIZE']}")


def compute_max_runtime(current_config):
    if current_config["APPROX_DATASET_SIZE"] >= 1_000_000:
        return 3 * SECONDS_IN_DAY
    elif current_config["APPROX_DATASET_SIZE"] >= 500_000:
        return 2 * SECONDS_IN_DAY
    else:
        return SECONDS_IN_DAY


def increase_verbosity_for_sagemaker(current_config):
    if current_config["IS_SAGEMAKER"]:
        current_config["VERBOSE"] = 2


def get_config():
    # creates a copy of the config
    current_config = _CONFIG.copy()

    # sets the sagemaker config
    increase_verbosity_for_sagemaker(current_config)

    # Updates the config based on the test run
    override_test_config(current_config)

    # sets the batch size based on the image size
    current_config["BATCH_SIZE"] = get_batch_size(current_config)

    # sets the max runtime based on the dataset size
    current_config["MaxRuntimeInSeconds"] = compute_max_runtime(current_config)

    return current_config


def get_path_config(pipeline_name):
    data_generation_output_prefix = "data_generation/output"
    training_output_prefix = "training/output"
    pipeline_path = f"s3://{S3_BUCKET}/{pipeline_name}"
    s3_data_generation_output_path = f"{pipeline_path}/{data_generation_output_prefix}"
    local_data_generation_output_path = f"{LOCAL_PATH_PREFIX}/{pipeline_name}/{data_generation_output_prefix}"
    # dict keys should be in caps
    path_config = {
        # INPUT RAW DATA S3 PATHS
        "S3_BUCKET": S3_BUCKET,
        "S3_DATA_GENERATION_OUTPUT_PATH": s3_data_generation_output_path,
        # ------------------------------------------------------------------ #
        # -------------------------- TRAINING PATHS ------------------------ #
        # ------------------------------------------------------------------ #
        # SAGEMAKER TRAINING LOCAL PATHS
        "SM_TRAINING_INPUT_PATH": "/opt/ml/input",
        "SM_TRAINING_OUTPUT_PATH": "/opt/ml/output",
        "SM_MODEL_DIR": "/opt/ml/output/model",
        "SM_CHECKPOINTS_PATH": "/opt/ml/checkpoints",
        # S3 CHECKPOINTS PATH
        "S3_TRAINING_CHECKPOINTS_PATH": f"{pipeline_path}/{training_output_prefix}/checkpoints",
    }

    if IS_SAGEMAKER:
        # --------------------- REMOTE DATA GENERATION STEP ---------------------- #
        path_config["INPUT_PATH"] = "/opt/ml/processing/input"
        path_config["OUTPUT_PATH"] = f"s3://{S3_BUCKET}/{S3_COMMON_PREFIX}/ai_or_not_datasets"
        path_config["CACHE_PATH"] = f"{path_config['INPUT_PATH']}/cache"
        path_config["INPUT_DATA_PATH"] = f"s3://{S3_BUCKET}/{S3_COMMON_PREFIX}/ai_or_not_datasets"
        path_config["TRAINING_DATASET_PATH"] = f"{s3_data_generation_output_path}/training_dataset"
        path_config["VALIDATION_DATASET_PATH"] = f"{s3_data_generation_output_path}/validation_dataset"
        path_config["TESTING_DATASET_PATH"] = f"{s3_data_generation_output_path}/testing_datasets"
    else:
        # --------------------- LOCAL DATA GENERATION STEP ---------------------- #
        path_config["INPUT_PATH"] = f"{LOCAL_PATH_PREFIX}/{pipeline_name}"
        path_config["OUTPUT_PATH"] = f"{local_data_generation_output_path}"
        path_config["CACHE_PATH"] = f"{path_config['OUTPUT_PATH']}/cache"
        path_config["INPUT_DATA_PATH"] = f"{LOCAL_PATH_PREFIX}/data"
        path_config["TRAINING_DATASET_PATH"] = f"{local_data_generation_output_path}/training_dataset"
        path_config["VALIDATION_DATASET_PATH"] = f"{local_data_generation_output_path}/validation_dataset"
        path_config["TESTING_DATASET_PATH"] = f"{local_data_generation_output_path}/testing_datasets"

    return path_config


def get_environment_variables():
    return {
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
    }


def setup_wandb(pipeline_name, environment, current_config, step_name):
    wandb_mode = "online" if IS_SAGEMAKER else "offline"
    os.environ["WANDB_MODE"] = wandb_mode
    run_name_suffix = pipeline_name.split("-")[-1].strip()

    return wandb.init(
        id=f"{run_name_suffix}_{step_name}",
        name=f"{run_name_suffix}_{step_name}",
        group=f"{run_name_suffix}",
        job_type=f"{step_name}_{current_config['MODEL_ARCHITECTURE']}",
        tags=[environment],
        settings=wandb.Settings(start_method="fork", init_timeout=120),
    )
