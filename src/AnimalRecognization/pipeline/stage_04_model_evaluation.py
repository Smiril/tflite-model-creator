from src.AnimalRecognization import logger
from src.AnimalRecognization.config.configuration import ConfigurationManager
from src.AnimalRecognization.components.model_evaluation import ModelEvaluation


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        metrics, report = model_evaluation.evaluate()

        # Log evaluation summary
        logger.info(f"Model evaluation metrics:")
        logger.info(f"Loss: {metrics['loss']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

        return metrics