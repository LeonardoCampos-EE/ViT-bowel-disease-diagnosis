import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class Dataloader(object):
    def __init__(self, csv: str, batch_size=8, input_shape=(224, 224, 3)) -> None:

        self.csv = pd.read_csv(csv)
        self.input_shape = input_shape
        self.batch_size = batch_size

    def build(self):

        self.train_csv, self.validation_csv = train_test_split(
            self.csv, test_size=0.1, stratify=self.csv.label
        )

        self.train_dataset_size = self.train_csv.shape[0]
        self.validation_dataset_size = self.validation_csv.shape[0]
        self.train_steps_per_epoch = self.train_csv.shape[0] // self.batch_size
        self.validation_steps_per_epoch = (
            self.validation_csv.shape[0] // self.batch_size
        )

        self.train_data_generator = ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=15,
            zoom_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
        )
        self.validation_data_generator = ImageDataGenerator(rescale=1 / 255.0)

        self.train_dataset = self.train_data_generator.flow_from_dataframe(
            dataframe=self.train_csv,
            x_col="path",
            y_col="label",
            subset="training",
            batch_size=self.batch_size,
            shuffle=True,
            class_mode="raw",
            target_size=(self.input_shape[0], self.input_shape[1]),
        )

        self.validation_dataset = self.validation_data_generator.flow_from_dataframe(
            dataframe=self.validation_csv,
            x_col="path",
            y_col="label",
            subset="validation",
            batch_size=self.batch_size,
            shuffle=True,
            class_mode="raw",
            target_size=(self.input_shape[0], self.input_shape[1]),
        )

        return self.train_dataset, self.validation_dataset
