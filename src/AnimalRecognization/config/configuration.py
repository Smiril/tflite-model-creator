(from pathlib import Path
from src.AnimalRecognization.utils.common import read_yaml, create_directories
from src.AnimalRecognization.entity.config_entity import (
    DataIngestionConfig,
    PrepareModelConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    PredictionConfig,
)
from src.AnimalRecognization.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

    def get_prepare_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_model
        create_directories([config.root_dir])

        return PrepareModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            params_image_size=self.params.base.image_size,
            params_batch_size=self.params.base.batch_size,
            params_learning_rate=self.params.model.learning_rate,
            params_num_classes=self.params.model.num_classes,
            params_dropout_rate=self.params.model.dropout_rate,
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        training = self.config.model_training
        prepare_model = self.config.prepare_model
        training_data = self.config.data_ingestion.unzip_dir

        create_directories([Path(training.root_dir)])

        return ModelTrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_model.model_path),
            training_data=Path(training_data),
            params_epochs=self.params.training.epochs,
            params_batch_size=self.params.base.batch_size,
            params_image_size=self.params.base.image_size,
            params_learning_rate=self.params.model.learning_rate,
            params_num_classes=self.params.model.num_classes,
            params_augmentation=self.params.training.augmentation,
            params_validation_split=self.params.training.validation_split,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        eval_config = self.config.model_evaluation
        model_training = self.config.model_training
        evaluation_data = self.config.data_ingestion.unzip_dir

        create_directories([Path(eval_config.root_dir)])

        return ModelEvaluationConfig(
            root_dir=Path(eval_config.root_dir),
            trained_model_path=Path(model_training.trained_model_path),
            evaluation_data=Path(evaluation_data),
            params_image_size=self.params.base.image_size,
            params_batch_size=self.params.base.batch_size,
            metrics_file_path=Path(eval_config.metrics_file_path),
            confusion_matrix_path=Path(eval_config.confusion_matrix_path),
        )

    def get_prediction_config(self) -> PredictionConfig:
        pred_config = self.config.prediction
        model_training = self.config.model_training

        create_directories([Path(pred_config.root_dir)])

        return PredictionConfig(
            root_dir=Path(pred_config.root_dir),
            trained_model_path=Path(model_training.trained_model_path),
            params_image_size=self.params.base.image_size,
            class_names_file=Path(pred_config.class_names_file),
            webapp_dir=Path(pred_config.webapp_dir),
        )
