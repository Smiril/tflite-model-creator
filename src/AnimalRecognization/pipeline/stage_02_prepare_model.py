from src.AnimalRecognization import logger
from src.AnimalRecognization.config.configuration import ConfigurationManager
from src.AnimalRecognization.components.prepare_model import PrepareModel


class PrepareModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.get_prepare_model_config()
        prepare_model = PrepareModel(config=prepare_model_config)
        model = prepare_model.prepare_model()