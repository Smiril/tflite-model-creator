from src.AnimalRecognization import logger
from src.AnimalRecognization.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.AnimalRecognization.pipeline.stage_02_prepare_model import PrepareModelTrainingPipeline
from src.AnimalRecognization.pipeline.stage_03_model_training import ModelTrainingPipeline
from src.AnimalRecognization.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from src.AnimalRecognization.pipeline.stage_05_prediction import PredictionPipeline

try:
   STAGE_NAME = "Data Ingestion stage"

   # Previous Stage 1 code
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

   # Stage 2
   STAGE_NAME = "Prepare Model stage"
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_model = PrepareModelTrainingPipeline()
   prepare_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

   # Stage 3
   STAGE_NAME = "Model Training stage"
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_training = ModelTrainingPipeline()
   model_training.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

   # Stage 4
   STAGE_NAME = "Model Evaluation stage"
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evaluation = ModelEvaluationPipeline()
   model_evaluation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

   STAGE_NAME = "Prediction Service setup"
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prediction = PredictionPipeline()
   prediction.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
   logger.exception(e)
   raise e