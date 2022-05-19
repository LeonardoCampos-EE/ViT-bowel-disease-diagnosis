import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
import math


class ShiftedPatchTokenization(layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        num_patches,
        projection_dim,
        epsilon,
        vanilla=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to swtich to vanilla patch extractor]
        self.epsilon = epsilon
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.projection_dim = projection_dim
        self.num_patches = num_patches

        self.flatten_patches = layers.Reshape((self.num_patches, -1))
        self.projection = layers.Dense(units=self.projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=self.epsilon)

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)

        return (tokens, patches)

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "vanilla": self.vanilla,
                "patch_size": self.patch_size,
                "epsilon": self.epsilon,
                "image_size": self.image_size,
                "half_patch": self.half_patch,
                "num_patches": self.num_patches,
            }
        )
        return config


class PositionalEncoder(layers.Layer):
    """
    This layer adds positional information to the encoded video tokens.
    """

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {"projection_dim": self.projection_dim, "num_patches": self.num_patches}
        )
        return config


class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The trainable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


class ModelBuilder:
    def __init__(
        self,
        num_classes: int = 3,
        input_shape: tuple = (224, 224, 3),
        num_layers: int = 8,
        num_heads: int = 8,
        projection_dim: int = 128,
        layer_normalization_eps: float = 1e-6,
        patch_size: int = 8,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.layer_normalization_eps = layer_normalization_eps
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_patches = (input_shape[0] // patch_size) ** 2

    def build_model(self, vanilla=False) -> keras.Model:
        input_layer = layers.Input(shape=self.input_shape)

        (tokens, _) = ShiftedPatchTokenization(
            image_size=self.input_shape[0],
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            projection_dim=self.projection_dim,
            epsilon=self.layer_normalization_eps,
            vanilla=vanilla,
        )(input_layer)
        encoded_patches = PositionalEncoder(self.num_patches, self.projection_dim)(
            tokens
        )

        diag_attn_mask = 1 - tf.eye(self.num_patches)
        diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

        # Create multiple layers of the Transformer block.
        for _ in range(self.num_layers):
            # Layer normalization and MHSA
            x1 = layers.LayerNormalization(epsilon=self.layer_normalization_eps)(
                encoded_patches
            )

            if not vanilla:
                attention_output = MultiHeadAttentionLSA(
                    num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
                )(x1, x1, attention_mask=diag_attn_mask)
            else:
                attention_output = layers.MultiHeadAttention(
                    num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
                )(x1, x1)
            # Skip connection
            x2 = layers.Add()([attention_output, encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=self.layer_normalization_eps)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=self.projection_dim * 2, activation=tf.nn.gelu),
                    layers.Dense(units=self.projection_dim, activation=tf.nn.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = layers.Add()([x3, x2])

        # Layer normalization and Global average pooling.
        representation = layers.LayerNormalization(
            epsilon=self.layer_normalization_eps
        )(encoded_patches)
        representation = layers.Flatten()(representation)

        # Classify outputs.
        output_layer = layers.Dense(units=self.num_classes, activation="softmax")(
            representation
        )

        # Create the Keras model.
        model = keras.Model(inputs=input_layer, outputs=output_layer)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,
        )

        model.compile(
            optimizer=optimizer,
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )

        return model
