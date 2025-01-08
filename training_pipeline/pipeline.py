import argparse
import json
import logging as log

import boto3
from config import PIPELINE_PREFIX
from config import RUN_NAME_SUFFIX
from config import get_config
from config import get_path_config
from sagemaker.inputs import TrainingInput
from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow import TensorFlowProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import TrainingStep

logger = log.getLogger()
logger.setLevel(log.INFO)


PIPELINE_NAME = f"{PIPELINE_PREFIX}-{RUN_NAME_SUFFIX}"
PATH_CONFIG = get_path_config(PIPELINE_NAME)
CURRNET_CONFIG = get_config()
ROLE = "arn:aws:iam::767087296931:role/service-role/AmazonSageMaker-ExecutionRole-20230227T093535"

COMPUTE_INSTANCE_TYPE = "ml.c5.2xlarge"  # 8 vCPUs, 16 GiB $0.408 USD/hour
GPU_INSTANCE_TYPE = "ml.p3.2xlarge"  # 8 vCPUs, 61 GiB, 1 GPU $3.825 USD/hour
VOLUME_SIZE = 1000


def create_processor_step(pipeline_session, environment):
    job_name = "processing-step"
    log.info(f"Running on instance type: {COMPUTE_INSTANCE_TYPE} with volume size: {VOLUME_SIZE}")

    data_processor = TensorFlowProcessor(
        base_job_name=job_name,
        framework_version="2.14",
        py_version="py310",
        role=ROLE,
        instance_type=COMPUTE_INSTANCE_TYPE,
        instance_count=1,
        volume_size_in_gb=VOLUME_SIZE,
        max_runtime_in_seconds=CURRNET_CONFIG["MaxRuntimeInSeconds"],
        sagemaker_session=pipeline_session,
    )

    run_args = data_processor.run(
        code="data_generator.py",
        source_dir=".",
        job_name=job_name,
        wait=False,
        arguments=["--pipeline_name", PIPELINE_NAME, "--environment", environment],
    )

    return ProcessingStep(name=job_name, step_args=run_args)


def get_training_step(pipeline_session, training_data_uri, environment):
    job_name = "training-step"

    classifier_estimator = TensorFlow(
        entry_point="train.py",
        role=ROLE,
        instance_count=1,
        instance_type=GPU_INSTANCE_TYPE,
        volume_size=VOLUME_SIZE,
        framework_version="2.14",
        py_version="py310",
        source_dir=".",
        max_run=CURRNET_CONFIG["MaxRuntimeInSeconds"],
        checkpoint_s3_uri=PATH_CONFIG["S3_TRAINING_CHECKPOINTS_PATH"],
        checkpoint_local_path=PATH_CONFIG["SM_CHECKPOINTS_PATH"],
        sagemaker_session=pipeline_session,
        hyperparameters={
            "pipeline_name": PIPELINE_NAME,
            "environment": environment,
        },
    )

    train_args = classifier_estimator.fit(
        job_name=job_name,
        inputs=TrainingInput(s3_data=training_data_uri, input_mode="FastFile"),
        wait=False,
    )

    return TrainingStep(name=job_name, step_args=train_args)


def get_pipeline_session(environment):
    default_bucket = PATH_CONFIG["S3_BUCKET"]
    default_bucket_prefix = "ai-image-code"
    if environment in ("dev", "development", "staging", "production"):
        boto_session = boto3.Session(profile_name=environment)
        return PipelineSession(
            boto_session=boto_session, default_bucket=default_bucket, default_bucket_prefix=default_bucket_prefix
        )
    else:
        boto_session = boto3.Session(profile_name="dev")
        return LocalPipelineSession(
            boto_session, default_bucket=default_bucket, default_bucket_prefix=default_bucket_prefix
        )


def get_training_data_uri(pipeline_session):
    if pipeline_session is LocalPipelineSession:
        return PATH_CONFIG["LOCAL_DATA_GENERATION_OUTPUT_PATH"]
    else:
        return PATH_CONFIG["S3_DATA_GENERATION_OUTPUT_PATH"]


def execute_pipeline(args):
    pipeline_session = get_pipeline_session(args.environment)
    log.info(f"pipeline_session: {pipeline_session}")
    processing_step = create_processor_step(pipeline_session, args.environment)
    training_step = get_training_step(
        pipeline_session,
        get_training_data_uri(pipeline_session),
        args.environment,
    )
    training_step.add_depends_on([processing_step])

    pipeline = Pipeline(
        name=f"{PIPELINE_PREFIX}",
        steps=[processing_step, training_step],
        sagemaker_session=pipeline_session,
    )

    log.info(f"Pipeline Definition: {json.loads(pipeline.definition())}")
    pipeline.upsert(role_arn=ROLE)

    execution = pipeline.start()

    # log execution details
    log.info(f"Pipeline Execution Details: {execution.describe()}")


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", help="Decides which AWS profile to use", default="local")

    return parser.parse_args()


if __name__ == "__main__":
    # parse args
    args = _parse_args()

    log.info("----------------- Training Config -----------------")
    log.info(f"Model Architecture: {CURRNET_CONFIG['MODEL_ARCHITECTURE']}")
    log.info(f"Image Size: {CURRNET_CONFIG['IM_SIZE']}")
    log.info(f"Batch Size: {CURRNET_CONFIG['BATCH_SIZE']}")
    log.info(f"Data uri: {PATH_CONFIG['S3_DATA_GENERATION_OUTPUT_PATH']}")
    log.info(f"Max Runtime: {int(CURRNET_CONFIG['MaxRuntimeInSeconds']/(60*60*24))} days")

    execute_pipeline(args)
