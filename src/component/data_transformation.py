import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTranformationConfig:
    preprocessor_file_path = os.path.join("artifacts","preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTranformationConfig()
        
    def get_transformer_object(self):
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course"
            ]
            
            # Create the Pipeline for numerical columns and categorical columns
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False)) # This not required for categorical
                ]
            )
            
            logging.info("Pipeline created for Numerical columns and Categorical columns")
            
            #Combine both pipeline for doing transformation
            preprocessor = ColumnTransformer(
                [
                    ("numerical_col_transform",num_pipeline,numerical_columns),
                    ("categorical_col_transform",cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def get_initiated_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data as dataframe completed")
            
            logging.info("Opening preprocessing object")
            preprocessor_obj=self.get_transformer_object()
            target_column = "math_score"
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course"
            ]
            
            #Drop the Target column for train and test dataset
            input_train_df=train_df.drop(columns=[target_column],axis=1)
            target_train_df = train_df[target_column]
            
            input_test_df=test_df.drop(columns=[target_column],axis=1)
            target_test_df = test_df[target_column]

            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            input_train_transformed_array=preprocessor_obj.fit_transform(input_train_df)
            input_test_transformed_array=preprocessor_obj.transform(input_test_df)
            
            #Concat the input array and target array of train data column wise 
            train_arr = np.c_[input_train_transformed_array,np.array(target_train_df)]
            test_arr = np.c_[input_test_transformed_array,np.array(target_test_df)]
            
            
            logging.info("Saving preprocessor object")
            
            #Save the transformed data into the pickle
            save_object(
                file_path = self.data_transform_config.preprocessor_file_path,
                obj = preprocessor_obj
            )
            
            return(
                   train_arr,
                   test_arr,
                   self.data_transform_config.preprocessor_file_path
                   )
            
            
        except Exception as e:
             raise CustomException(e,sys)
        
            


