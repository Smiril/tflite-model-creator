from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    model_path: Path
    params_image_size: list
    params_batch_size: int
    params_learning_rate: float
    params_num_classes: int
    params_dropout_rate: float

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_image_size: list
    params_learning_rate: float
    params_num_classes: int
    params_augmentation: bool
    params_validation_split: float

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    trained_model_path: Path
    evaluation_data: Path
    params_image_size: list
    params_batch_size: int
    metrics_file_path: Path
    confusion_matrix_path: Path

@dataclass(frozen=True)
class PredictionConfig:
    root_dir: Path
    trained_model_path: Path
    params_image_size: list
    class_names_file: Path
    webapp_dir: Path