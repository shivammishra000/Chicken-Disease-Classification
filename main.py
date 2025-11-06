from chickenDiseaseClassification import logger
from chickenDiseaseClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from chickenDiseaseClassification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
# from chickenDiseaseClassification.pipeline.stage_03_training import ModelTrainingPipeline
# from chickenDiseaseClassification.pipeline.stage_04_evaluation import EvaluationPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e