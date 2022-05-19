import argparse
import os

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

from src.callbacks import CallbackBuilder
from src.dataloader import Dataloader
from src.model import ModelBuilder


class Trainer:
    def __init__(
        self,
        dataset_csv: str,
        output_dir: str,
        batch_size: int,
        epochs: int,
    ) -> None:

        self.dataset_csv = dataset_csv
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.batch_size = batch_size
        self.epochs = epochs

        return

    def build_datasets(self) -> None:

        self.dataloader = Dataloader(self.dataset_csv, self.batch_size)
        self.train_dataset, self.validation_dataset = self.dataloader.build()
        return

    def build_callbacks(self) -> None:

        callback_builder = CallbackBuilder(self.output_dir)
        self.callbacks = callback_builder.build()

        return

    def build_model(self) -> None:

        # Hyper parameters
        num_classes = 3

        model_builder = ModelBuilder(
            num_classes=num_classes,
            input_shape=(224, 224, 3),
            num_heads=8,
            projection_dim=128,
            patch_size=8,
        )
        self.model = model_builder.build_model()

        return

    def get_class_weights(self) -> None:

        self.class_weights = class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(self.dataloader.train_csv.label),
            y=self.dataloader.train_csv.label,
        )
        self.class_weights = dict(enumerate(self.class_weights))

        return

    def train(self) -> None:

        self.build_datasets()
        self.build_callbacks()
        self.build_model()
        self.get_class_weights()

        self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            verbose=1,
            steps_per_epoch=self.dataloader.train_steps_per_epoch,
            validation_steps=self.dataloader.validation_steps_per_epoch,
            callbacks=self.callbacks,
            validation_data=self.validation_dataset,
            use_multiprocessing=True,
            workers=8,
            class_weight=self.class_weights,
        )

        tf.keras.backend.clear_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Size of each batch of clips"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to the dataset CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vit",
        help="Path to the directory where the trained model will be saved",
    )
    args = parser.parse_args()

    trainer = Trainer(
        args.csv,
        args.output_dir,
        args.batch_size,
        args.epochs,
    )
    trainer.train()
