import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.component.data_transformation import DataTranformationConfig,DataTransformation
from src.component.model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            #Here we write logic to bring the data from database or mongodb or from any file
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info("Data has been read as dataframe")
            
            #directory "artifacts" creating under mlproject-env
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)
            
            logging.info("train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            
            test_set.to_csv(self.ingestion_config.test_data_path,header=True,index=False)
            
            logging.info("Train test split completed")
            
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
            
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    data=DataIngestion()
    train_data_path,test_data_path = data.initiate_data_ingestion()
    
    data_transform=DataTransformation()
    train_arr,test_rr,preprocessor_path=data_transform.get_initiated_data_transformation(train_data_path,test_data_path)


    model_trainer=ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_rr,preprocessor_path)
    
