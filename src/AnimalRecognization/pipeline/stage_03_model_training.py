from src.AnimalRecognization import logger
from src.AnimalRecognization.config.configuration import ConfigurationManager
from src.AnimalRecognization.components.model_training import ModelTraining


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model_training.train()