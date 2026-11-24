import os
import subprocess
import numpy as np
import tensorflow as tf
import tensorflow.lite as lite
import tf2onnx
import coremltools as ct
import coremltools.optimize.coreml as cto
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.calib_dir = None

    def _find_dataset_path(self):
        """Findet das Verzeichnis mit den Klassenordnern (oder train/validate Unterordner)."""
        base_path = self.config.training_data
        nested_paths = [
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals", "animals"),
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals"),
            base_path,
        ]

        for path in nested_paths:
            if os.path.exists(path) and len([
                d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
            ]) > 0:
                logger.info(f"Verwende Trainingsdaten unter: {path}")
                return path

        raise ValueError("Kein gültiges Trainingsdaten-Verzeichnis gefunden")

    def save_labels_from_generator(self, generator, output_file):
        """Speichert die Klassen in der exakten Reihenfolge des Keras-Generators (0..N)."""
        sorted_classes = sorted(generator.class_indices.items(), key=lambda x: x[1])
        with open(output_file, "w", encoding="utf-8") as f:
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")
        logger.info(f"Labels gespeichert unter: {output_file}")

    def get_image_dataset(self):
        """Erstellt Trainings- und Validierungs-Generatoren."""
        dataset_path = self._find_dataset_path()
        train_subdir = os.path.join(dataset_path, "train")
        val_subdir = os.path.join(dataset_path, "validate")
        has_explicit_split = os.path.isdir(train_subdir) and os.path.isdir(val_subdir)

        if self.config.params_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode="nearest",
                brightness_range=[0.8, 1.2],
                validation_split=0.0 if has_explicit_split else self.config.params_validation_split,
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                validation_split=0.0 if has_explicit_split else self.config.params_validation_split,
            )

        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.0 if has_explicit_split else self.config.params_validation_split,
        )

        if has_explicit_split:
            train_generator = train_datagen.flow_from_directory(
                train_subdir,
                target_size=tuple(self.config.params_image_size),
                batch_size=self.config.params_batch_size,
                class_mode="categorical",
                shuffle=True,
            )
            val_generator = val_datagen.flow_from_directory(
                val_subdir,
                target_size=tuple(self.config.params_image_size),
                batch_size=self.config.params_batch_size,
                class_mode="categorical",
                shuffle=False,
            )
            self.calib_dir = val_subdir
        else:
            seed = 42
            train_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=tuple(self.config.params_image_size),
                batch_size=self.config.params_batch_size,
                class_mode="categorical",
                subset="training",
                shuffle=True,
                seed=seed,
            )
            val_generator = val_datagen.flow_from_directory(
                dataset_path,
                target_size=tuple(self.config.params_image_size),
                batch_size=self.config.params_batch_size,
                class_mode="categorical",
                subset="validation",
                shuffle=False,
                seed=seed,
            )
            self.calib_dir = dataset_path

        labels_path = os.path.join(self.config.root_dir, "labels.txt")
        self.save_labels_from_generator(train_generator, labels_path)

        class_weights = self._calculate_class_weights(train_generator.classes)
        logger.info(f"Training: {train_generator.samples} Bilder in {train_generator.num_classes} Klassen")
        logger.info(f"Validierung: {val_generator.samples} Bilder")

        return train_generator, val_generator, class_weights

    def _calculate_class_weights(self, classes):
        unique_classes = np.unique(classes)
        if len(unique_classes) <= 1:
            return {}
        try:
            weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=unique_classes, y=classes
            )
            return {i: w for i, w in enumerate(weights)}
        except Exception as e:
            logger.warning(f"Class-Weights konnten nicht berechnet werden ({e})")
            return {}

    def validate_dataset(self):
        """Prüft alle Bilder und entfernt beschädigte Dateien."""
        dataset_path = self._find_dataset_path()
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        problem_files = []

        for root, _, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                if ext not in valid_extensions:
                    problem_files.append(file_path)
                    continue
                try:
                    raw_bytes = tf.io.read_file(file_path)
                    tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
                except Exception:
                    problem_files.append(file_path)

        for path in problem_files:
            try:
                os.remove(path)
                logger.info(f"Beschädigte Datei entfernt: {path}")
            except Exception as e:
                logger.warning(f"Konnte {path} nicht entfernen: {e}")

    def _create_callbacks(self):
        return [
            ModelCheckpoint(
                filepath=os.path.join(self.config.root_dir, "best_model.h5"),
                save_best_only=True,
                monitor="val_accuracy",
                verbose=1,
            ),
            EarlyStopping(
                patience=7,
                restore_best_weights=True,
                monitor="val_loss",
                verbose=1,
            ),
            ReduceLROnPlateau(
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                monitor="val_loss",
                verbose=1,
            ),
            TensorBoard(
                log_dir=os.path.join(self.config.root_dir, "logs"),
                update_freq="epoch",
            ),
        ]

    def _get_base_model(self, model):
        for layer in model.layers:
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                return layer
        return None

    def _compile_model(self, model, lr):
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def export_edge_models(self, model, val_generator):
        """Exportiert das trainierte Modell für TFLite, EdgeTPU, CoreML und Axelera AIPU."""
        img_h, img_w = tuple(self.config.params_image_size)
        base_export_path = os.path.join(self.config.root_dir, "model.tflite")

        # 1. Standard TFLite (Float32) -> für --device cpu & --device gpu
        logger.info("[1/4] Berechne model.tflite (Float32)...")
        converter_fp32 = lite.TFLiteConverter.from_keras_model(model)
        tflite_fp32_model = converter_fp32.convert()
        with open(base_export_path, "wb") as f:
            f.write(tflite_fp32_model)
        logger.info(f"Gespeichert: {base_export_path}")

        # 2. Full-Integer Int8 TFLite -> für --device tpu (Coral Edge TPU)
        logger.info("[2/4] Berechne Int8 TFLite für Edge TPU...")

        def representative_dataset():
            val_generator.reset()
            steps = min(100, len(val_generator))
            for _ in range(steps):
                images, _ = next(val_generator)
                for i in range(images.shape[0]):
                    yield [np.expand_dims(images[i], axis=0).astype(np.float32)]

        int8_tflite_path = base_export_path.replace(".tflite", "_edgetpu_int8.tflite")
        converter_int8 = lite.TFLiteConverter.from_keras_model(model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int8.representative_dataset = representative_dataset
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.uint8
        converter_int8.inference_output_type = tf.uint8

        tflite_int8_model = converter_int8.convert()
        with open(int8_tflite_path, "wb") as f:
            f.write(tflite_int8_model)
        logger.info(f"Gespeichert: {int8_tflite_path}")

        try:
			subprocess.run(["edgetpu_compiler", int8_tflite_path, "-o", self.config.root_dir], check=True)
			logger.info("EdgeTPU-Kompilierung erfolgreich abgeschlossen.")
		except (FileNotFoundError, subprocess.CalledProcessError):
			logger.info(
				f"Hinweis: edgetpu_compiler nicht gefunden. Auf Linux ausführen: edgetpu_compiler {int8_tflite_path}"
			)
		# 3. CoreML .mlpackage (Int8) -> für --device mps & --device npu
		logger.info("[3/4] Berechne CoreML .mlpackage (Int8) für Apple Silicon...")
		coreml_output_path = base_export_path.replace(".tflite", ".mlpackage")
		mlmodel = ct.convert(
				model,
				inputs=[ct.TensorType(name="input_1", shape=(1, img_h, img_w, 3))],
				compute_units=ct.ComputeUnit.ALL,
				minimum_deployment_target=ct.target.macOS13,
				)
		op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
		opt_config = cto.OptimizationConfig(global_config=op_config)
		mlmodel_int8 = cto.linear_quantize_weights(mlmodel, config=opt_config)
		mlmodel_int8.save(coreml_output_path)
		logger.info(f"Gespeichert: {coreml_output_path}")
		# 4. Axelera AI Metis AIPU (.onnx -> .axm) -> für --device axelera
		logger.info("[4/4] Exportiere ONNX und kompiliere für Axelera Metis AIPU...")
		onnx_output_path = base_export_path.replace(".tflite", ".onnx")
		axm_output_path = base_export_path.replace(".tflite", ".axm")
		spec = (tf.TensorSpec((1, img_h, img_w, 3), tf.float32, name="input_1"),)
		tf2onnx.convert.from_keras(
				model, input_signature=spec, opset=15, output_path=onnx_output_path
				)
		logger.info(f"ONNX-Modell gespeichert: {onnx_output_path}")
		try:
			from axelera.compiler import compile_model
				compile_model(
				model_path=onnx_output_path,
				output_path=axm_output_path,
				calibration_dataset=self.calib_dir,
				)
				logger.info(f"Axelera AIPU-Modell gespeichert: {axm_output_path}")
		except ImportError:
				logger.info(
							"Hinweis: axelera.compiler nicht gefunden. Auf Axelera-Host ausführen:\n"
							f"  axcompiler --model {onnx_output_path} --output {axm_output_path} --quantize"
							)
		def train(self):
			"""Trainiert das MobileNetV2 .h5 Modell und exportiert alle Zielformate."""
			self.validate_dataset()
			logger.info(f"Lade .h5 Basismodell von {self.config.updated_base_model_path}")
			model = tf.keras.models.load_model(str(self.config.updated_base_model_path), compile=False)
			base_model = self._get_base_model(model)
			if base_model is not None:
				for layer in model.layers:
					layer.trainable = True
					base_model.trainable = False
					train_generator, val_generator, class_weights = self.get_image_dataset()
					callbacks = self._create_callbacks()
					steps_per_epoch = max(1, train_generator.samples // self.config.params_batch_size)
					validation_steps = max(1, val_generator.samples // self.config.params_batch_size)
					# Phase 1: Nur Classification Head trainieren
					logger.info("Phase 1: Training des Classification Heads")
					initial_epochs = max(2, self.config.params_epochs // 3)
					self._compile_model(model, lr=self.config.params_learning_rate)
					history = model.fit(
							train_generator,
							steps_per_epoch=steps_per_epoch,
							epochs=initial_epochs,
							validation_data=val_generator,
							validation_steps=validation_steps,
							callbacks=callbacks,
							class_weight=class_weights,
							verbose=1,
							)
					# Phase 2: Fine-Tuning der obersten MobileNetV2-Blöcke
					logger.info("Phase 2: Fine-Tuning der oberen MobileNetV2-Layer")
					if base_model is not None:
						base_model.trainable = True
						for layer in base_model.layers[:-35]:
							layer.trainable = False
							for layer in base_model.layers[-35:]:
								if isinstance(layer, tf.keras.layers.BatchNormalization):
									layer.trainable = False
					else:
						layer.trainable = True
						self._compile_model(model, lr=self.config.params_learning_rate / 100)
						last_epoch = history.epoch[-1] + 1 if history.epoch else initial_epochs
						history = model.fit(
								train_generator,
								steps_per_epoch=steps_per_epoch,
								epochs=self.config.params_epochs,
								initial_epoch=last_epoch,
								validation_data=val_generator,
								validation_steps=validation_steps,
								callbacks=callbacks,
								class_weight=class_weights,
								verbose=1,
						)
					best_model_path = os.path.join(self.config.root_dir, "best_model.h5")
					if os.path.exists(best_model_path):
						logger.info(f"Lade bestes Checkpoint-Modell: {best_model_path}")
						model = tf.keras.models.load_model(best_model_path)
						model.save(str(self.config.trained_model_path))
						logger.info(f"Finales .h5 Modell gespeichert: {self.config.trained_model_path}")
						self.export_edge_models(model, val_generator)
			
            return history
