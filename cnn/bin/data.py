import os
import argparse
import tensorflow as tf
from pathlib import Path


def make_datasets(dataset_dir, image_size=224, batch_size=32, seed=42):
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    train_ds_full = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    # save class names before splitting
    class_names = train_ds_full.class_names

    # split into train/validation
    val_size = 0.2
    total_batches = tf.data.experimental.cardinality(train_ds_full).numpy()
    val_batches = int(total_batches * val_size)

    val_ds = train_ds_full.take(val_batches)
    train_ds = train_ds_full.skip(val_batches)

    # load test set
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    # pptimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    print(f"\n===> Dataset ready — Classes: {class_names}")
    return train_ds, val_ds, test_ds, class_names
