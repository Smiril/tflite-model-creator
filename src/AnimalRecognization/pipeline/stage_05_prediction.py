from src.AnimalRecognization import logger
from src.AnimalRecognization.config.configuration import ConfigurationManager
from src.AnimalRecognization.components.prediction import PredictionService


class PredictionPipeline:
    def __init__(self):
        self.prediction_service = None

    def get_prediction_service(self):
        """Initialize and return prediction service"""
        if self.prediction_service is None:
            config = ConfigurationManager()
            prediction_config = config.get_prediction_config()
            self.prediction_service = PredictionService(config=prediction_config)

            # Load model and class names
            self.prediction_service.load_model()
            self.prediction_service.load_class_names()

        return self.prediction_service

    def main(self):
        """Set up prediction service and webapp"""
        logger.info("Initializing prediction service")
        prediction_service = self.get_prediction_service()

        # Create Flask app
        prediction_service.setup_flask_app()

        logger.info("Prediction service setup complete")
        logger.info(f"Run 'python {prediction_service.config.webapp_dir}/app.py' to start the web application")