import os
import argparse
import tensorflow as tf
from pathlib import Path


def build_model(num_classes, image_size=224):
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=(image_size, image_size, 3), weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model
