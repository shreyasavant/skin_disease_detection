import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models, applications, callbacks
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def plot_training_curves(history, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    pdf_path = os.path.join(output_dir, "training_curves_v" + timestamp_str + ".pdf")

    with PdfPages(pdf_path) as pdf:
        # accuracy plot
        plt.figure(figsize=(8, 6))
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        pdf.savefig()
        plt.close()

        # loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        pdf.savefig()
        plt.close()

    print(f"\n===> Training curves saved to {pdf_path}")


def train_model(model, base_model, train_ds, val_ds, epochs, model_out):
    ckpt_cb = callbacks.ModelCheckpoint(
        filepath="models/cnn_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_cb = callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )

    print("\n===> Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt_cb, early_cb, reduce_lr],
    )

    print("\n===> Generating training performance plots...")
    plot_training_curves(history)

    print("\n===> Fine-tuning the base model...")
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * 0.6)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_epochs = max(3, int(epochs * 0.3))
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=[ckpt_cb, early_cb, reduce_lr],
    )

    print("\n===> Adding fine-tuning performance to training report...")
    plot_training_curves(fine_history)

    if not model_out.endswith(".keras"):
        model_out = os.path.splitext(model_out)[0] + ".keras"

    model.save(model_out)
    print(f"===> Model saved to {model_out}")

    return history
