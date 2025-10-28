import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def load_test_dataset(dataset_dir, image_size=224, batch_size=32):
    test_dir = os.path.join(dataset_dir, "test")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = test_ds.class_names
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return test_ds, class_names


def evaluate_model(model_path, dataset_dir, image_size=224, batch_size=32):
    print(f"\n===> Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"\n===> Loading test dataset from: {dataset_dir}/test")
    test_ds, class_names = load_test_dataset(dataset_dir, image_size, batch_size)

    print("\n===> Evaluating model on test dataset...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_prob = model.predict(test_ds)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n===> Evaluation Complete!")
    print(
        f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}"
    )

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n===> Classification Report:\n", report)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # create results folder
    os.makedirs("results", exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    pdf_path = os.path.join("results", "model_evaluation_v" + timestamp_str + ".pdf")

    with PdfPages(pdf_path) as pdf:
        # confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        pdf.savefig()
        plt.close()

        # metrics summary page
        plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        text = (
            f"Model Evaluation Summary\n\n"
            f"Accuracy: {acc:.4f}\n"
            f"Precision: {prec:.4f}\n"
            f"Recall: {rec:.4f}\n"
            f"F1-score: {f1:.4f}\n\n"
            f"Classification Report:\n{report}"
        )
        plt.text(0.02, 0.98, text, fontsize=10, va="top", ha="left", family="monospace")
        pdf.savefig()
        plt.close()

    print(f"\n===> Evaluation report saved to: {pdf_path}")
