import os
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


class CallbackBuilder:
    def __init__(self, log_path: str):
        self.log_path = log_path
        return

    def build(self):
        # Build tensorboard callback
        tensorboard_callback = TensorBoard(
            log_dir=os.path.join(self.log_path, "tensorboard"),
            histogram_freq=0,
            write_graph=False,
            write_images=False,
        )

        # Build checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.log_path, "model.{epoch:d}.hdf5"),
            monitor="val_accuracy",
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode="max",
            options=None,
            period=1,
        )

        early_stop = EarlyStopping(monitor="val_accuracy", patience=50)

        callbacks = [tensorboard_callback, checkpoint_callback]

        return callbacks
