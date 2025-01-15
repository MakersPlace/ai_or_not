import logging as log

import numpy as np
import pandas as pd
import tensorflow as tf
from data_processing.data_loader import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import wandb


def get_multi_label_metrics(multi_label_true_labels, multi_label_predicted_logits):
    sparse_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    multi_label_loss = sparse_loss(multi_label_true_labels, multi_label_predicted_logits).numpy()

    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    acc_metric.update_state(multi_label_true_labels, multi_label_predicted_logits)
    multi_label_accuracy = acc_metric.result().numpy()

    log.info(f"Multi Label Loss: {multi_label_loss} Multi Label Acc: {multi_label_accuracy}")

    return multi_label_loss, multi_label_accuracy


def get_multi_label_logits(current_dataset, current_model, config):
    # Extract image from current_dataset as numpy array
    predicted_logits = []
    true_labels = []

    for images, labels in current_dataset.as_numpy_iterator():
        predicted_logits.extend(current_model.predict_on_batch(images))
        true_labels.extend(labels)

    true_labels = np.array(true_labels, dtype=np.int32)
    predicted_logits = np.array(predicted_logits, dtype=np.float32)
    log.info(f"True Labels Shape: {true_labels.shape} Predicted Logits Shape: {predicted_logits.shape}")

    return true_labels, predicted_logits


def get_multi_label_predictions(predicted_logits):
    predicted_labels = np.argmax(predicted_logits, axis=1)
    predicted_labels = predicted_labels.astype(np.int32)

    return predicted_labels


def get_binary_predictions(multi_label_predicted_labels, multi_label_true_labels):
    binary_predicted_labels = np.clip(multi_label_predicted_labels, 0, 1)
    binary_true_labels = np.clip(multi_label_true_labels, 0, 1)

    return binary_predicted_labels, binary_true_labels


def get_binary_acc(binary_predicted_labels, true_labels):
    # caclulate loss and accuracy for binary classification using scikit-learn
    binary_accuracy = accuracy_score(true_labels, binary_predicted_labels)

    return binary_accuracy


def log_classification_report(predicted_labels, true_labels, class_names, wandb_run):
    if len(class_names) == 2:
        classification_report_name = "Binary Classification Matrix"
        label_report_name = "Binary Label Report"
    else:
        classification_report_name = "Multi Label Classification Matrix"
        label_report_name = "Multi Label Label Report"

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
    )
    # print report to console
    log.info(f"{classification_report_name}: {report}")
    # count the number of samples in each class
    true_label_counts = np.unique(true_labels, return_counts=True)
    predicted_label_counts = np.unique(predicted_labels, return_counts=True)

    log.info(f"True label counts: {true_label_counts}")
    log.info(f"Predicted label counts: {predicted_label_counts}")

    avg_report_df = pd.DataFrame()
    avg_report_df["model"] = [wandb_run.name]
    avg_report_df["Avg precision"] = [report["macro avg"]["precision"]]
    avg_report_df["Avg recall"] = [report["macro avg"]["recall"]]
    avg_report_df["Avg F1"] = [report["macro avg"]["f1-score"]]
    avg_report_df["Avg accuracy"] = [report["accuracy"]]
    avg_report_df["Avg support"] = [report["macro avg"]["support"]]

    label_reoprt_df = pd.DataFrame()
    label_reoprt_df["model"] = [wandb_run.name]
    for class_name in class_names:
        label_reoprt_df[f"{class_name} P/R/F1"] = (
            f"{report[class_name]['precision']:.2f} /"
            + f" {report[class_name]['recall']:.2f} /"
            + f" {report[class_name]['f1-score']:.2f}"
        )

    log.info(f"{classification_report_name}: {report}")
    log.info(f"{label_report_name}: {label_reoprt_df}")

    wandb_run.log({classification_report_name: wandb.Table(dataframe=avg_report_df)})
    wandb_run.log({label_report_name: wandb.Table(dataframe=label_reoprt_df)})


def log_confusion_matrix(dataset_name, predicted_labels, true_labels, class_names, wandb_run):
    if len(class_names) == 2:
        plot_name = "Binary Confusion Matrix"
    else:
        plot_name = "Multi Label Confusion Matrix"
    plot_name = f"{dataset_name} {plot_name}"

    wandb_run.log(
        {
            plot_name: wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_labels,
                preds=predicted_labels,
                class_names=class_names,
                title=plot_name,
            )
        }
    )


def log_individual_dataset_metrics(current_model, config, wandb_run):
    data_loader = DataLoader(config)
    binary_accuracy_dataframe = pd.DataFrame()
    multi_label_accuracy_dataframe = pd.DataFrame()
    multi_label_loss_dataframe = pd.DataFrame()
    counts_dataframe = pd.DataFrame()

    testing_datasets = config["TESTING_DATASET_PATH"]
    testing_datasets = tf.io.gfile.glob(f"{testing_datasets}/*")

    all_true_labels = None
    all_predicted_logits = None

    # walk through the testing_datasets directory and get all the subdirectories
    for directory in testing_datasets:
        if not tf.io.gfile.isdir(directory):
            log.error(f"Invalid Directory: {directory}")
            continue

        dataset_name = directory.split("/")[-1]
        log.info(f"Testing Dataset {dataset_name}")

        current_dataset = data_loader.load_testing_dataset(directory)
        if config["TEST_RUN"]:
            # shard the dataset to get a smaller sample
            current_dataset = current_dataset.shard(10, 0)

        multi_label_true_labels, multi_label_predicted_logits = get_multi_label_logits(
            current_dataset, current_model, config
        )
        multi_label_predicted_labels = get_multi_label_predictions(multi_label_predicted_logits)
        binary_predicted_labels, binary_true_labels = get_binary_predictions(
            multi_label_predicted_labels, multi_label_true_labels
        )
        test_accuracy = get_binary_acc(binary_predicted_labels, binary_true_labels)

        multi_label_loss, multi_label_accuracy = get_multi_label_metrics(
            multi_label_true_labels, multi_label_predicted_logits
        )

        binary_accuracy_dataframe[dataset_name] = [test_accuracy]
        multi_label_accuracy_dataframe[dataset_name] = [multi_label_accuracy]
        multi_label_loss_dataframe[dataset_name] = [multi_label_loss]
        counts_dataframe[dataset_name] = [len(binary_true_labels)]

        log.info(f"{dataset_name} - Binary Acc: {test_accuracy:.4f} Multi Label Acc: {multi_label_accuracy:.4f}")

        log_confusion_matrix(
            dataset_name, multi_label_predicted_labels, multi_label_true_labels, config["CLASS_NAMES"], wandb_run
        )

        all_true_labels = (
            np.concatenate((all_true_labels, multi_label_true_labels))
            if all_true_labels is not None
            else multi_label_true_labels
        )
        all_predicted_logits = (
            np.concatenate((all_predicted_logits, multi_label_predicted_logits))
            if all_predicted_logits is not None
            else multi_label_predicted_logits
        )

    wandb_run.log({"Binary Accuracy": wandb.Table(dataframe=binary_accuracy_dataframe)})
    wandb_run.log({"Multi Label Accuracy": wandb.Table(dataframe=multi_label_accuracy_dataframe)})
    wandb_run.log({"Multi Label Loss": wandb.Table(dataframe=multi_label_loss_dataframe)})
    wandb_run.log({"Test Dataset Size": wandb.Table(dataframe=counts_dataframe)})

    return all_true_labels, all_predicted_logits


def log_composite_dataset_metrics(multi_label_true_labels, multi_label_predicted_logits, config, wandb_run):
    composite_dataframe = pd.DataFrame()
    log.info("Testing All Test datasets Together")

    multi_label_predicted_labels = get_multi_label_predictions(multi_label_predicted_logits)
    binary_predicted_labels, binary_true_labels = get_binary_predictions(
        multi_label_predicted_labels, multi_label_true_labels
    )
    composite_accuracy = get_binary_acc(binary_predicted_labels, binary_true_labels)

    multi_label_loss, multi_label_accuracy = get_multi_label_metrics(
        multi_label_true_labels, multi_label_predicted_logits
    )

    composite_dataframe["model"] = [wandb_run.name]
    composite_dataframe["Binary accuracy"] = [composite_accuracy]
    composite_dataframe["Multi label accuracy"] = [multi_label_accuracy]
    composite_dataframe["Multi label loss"] = [multi_label_loss]
    log.info(f"All Test datasets - Binary Acc: {composite_accuracy:.4f} Multi Label Acc: {multi_label_accuracy:.4f}")
    wandb_run.log({"Overall Results": wandb.Table(dataframe=composite_dataframe)})

    log_classification_report(binary_predicted_labels, binary_true_labels, config["BINARY_CLASS_NAMES"], wandb_run)
    log_confusion_matrix(
        "Overall", binary_predicted_labels, binary_true_labels, config["BINARY_CLASS_NAMES"], wandb_run
    )

    log_classification_report(multi_label_predicted_labels, multi_label_true_labels, config["CLASS_NAMES"], wandb_run)
    log_confusion_matrix(
        "Overall", multi_label_predicted_labels, multi_label_true_labels, config["CLASS_NAMES"], wandb_run
    )


def evaluate_model(current_model, config, wandb_run):
    current_model.trainable = False

    all_true_labels, all_predicted_logits = log_individual_dataset_metrics(
        current_model,
        config,
        wandb_run,
    )

    log_composite_dataset_metrics(
        all_true_labels,
        all_predicted_logits,
        config,
        wandb_run,
    )
