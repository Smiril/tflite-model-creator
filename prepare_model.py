import tensorflow as tf
from pathlib import Path
from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import PrepareModelConfig


class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config

    def get_base_model(self):
        """Erstellt das vortrainierte MobileNetV2 Basismodell."""
        input_shape = list(self.config.params_image_size) +

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        logger.info(f"MobileNetV2 Basismodell geladen (Input: {input_shape})")
        return base_model

    def prepare_full_model(self, base_model):
        """Fügt Rescaling (für TFLite/CoreML/ONNX) und Classification Head hinzu."""
        input_shape = list(self.config.params_image_size) +
        inputs = tf.keras.Input(shape=input_shape, name="input_1")

        # Wandelt Bilder aus ImageDataGenerator(rescale=1./255) exakt in [-1, 1] für MobileNetV2 um.
        # Vorteil: TFLite Int8, CoreML und ONNX können direkt mit [0..1] bzw. uint8 [0..255] arbeiten.
        x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)

        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Classification Head (identisch zur Struktur deines Export-Skripts)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.config.params_dropout_rate)(x)
        outputs = tf.keras.layers.Dense(
            self.config.params_num_classes, activation="softmax", name="predictions"
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info("Vollständiges MobileNetV2 Modell vorbereitet")
        return model

    def save_model(self, path: Path, model: tf.keras.Model):
        model.save(str(path))
        logger.info(f"Modell gespeichert unter: {path}")

    def prepare_model(self):
        base_model = self.get_base_model()
        full_model = self.prepare_full_model(base_model)
        self.save_model(self.config.model_path, full_model)
        return full_model
