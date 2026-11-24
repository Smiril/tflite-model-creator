import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def get_test_dataset(self):
        """Create test/validation dataset matching MobileNetV2 model preprocessing"""
        base_path = self.config.evaluation_data
        nested_paths = [
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals", "animals"),
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals"),
            base_path
        ]

        test_path = None
        for path in nested_paths:
            if os.path.exists(path) and len([
                d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
            ]) > 0:
                test_path = path
                break

        if not test_path:
            raise ValueError("Could not find valid test data directory")

        # Prüfen, ob ein expliziter 'validate'-Unterordner existiert
        val_subdir = os.path.join(test_path, "validate")
        has_explicit_split = os.path.isdir(os.path.join(test_path, "train")) and os.path.isdir(val_subdir)

        if has_explicit_split:
            logger.info(f"Using explicit validation directory at: {val_subdir}")
            # Modell enthält Rescaling(scale=2.0, offset=-1.0), daher bleibt rescale=1./255 exakt korrekt
            test_datagen = ImageDataGenerator(rescale=1.0 / 255)
            test_generator = test_datagen.flow_from_directory(
                val_subdir,
                target_size=tuple(self.config.params_image_size),
                batch_size=self.config.params_batch_size,
                class_mode="categorical",
                shuffle=False
            )
        else:
            logger.info(f"Using validation split (seed=42) at: {test_path}")
            # Identischer Split & Seed wie in ModelTraining, um Trainingsdaten auszuschließen
            test_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
            test_generator = test_datagen.flow_from_directory(
                test_path,
                target_size=tuple(self.config.params_image_size),
                batch_size=self.config.params_batch_size,
                class_mode="categorical",
                subset="validation",
                shuffle=False,
                seed=42
            )

        logger.info(f"Found {test_generator.samples} evaluation images in {test_generator.num_classes} classes")
        return test_generator

    def evaluate(self):
        """Evaluate the trained MobileNetV2 .h5 model on test/validation data"""
        logger.info(f"Loading .h5 model from {self.config.trained_model_path}")
        model = tf.keras.models.load_model(str(self.config.trained_model_path))

        test_generator = self.get_test_dataset()

        logger.info("Evaluating model...")
        scores = model.evaluate(test_generator, verbose=1)
        metrics = {"loss": float(scores[0]), "accuracy": float(scores[1])}

        logger.info("Generating predictions for confusion matrix...")
        y_pred_probs = model.predict(test_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_generator.classes

        # Classification Report als CSV speichern
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()

        metrics_file = self.config.metrics_file_path
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        report_df.to_csv(metrics_file)
        logger.info(f"Classification report saved to {metrics_file}")

        # Confusion Matrix speichern
        os.makedirs(os.path.dirname(self.config.confusion_matrix_path), exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(24, 24))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (MobileNetV2)")
        plt.tight_layout()
        plt.savefig(self.config.confusion_matrix_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {self.config.confusion_matrix_path}")

        return metrics
