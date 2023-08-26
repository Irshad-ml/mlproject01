import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path ="artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            scaled_custom_data = preprocessor.transform(features)
            prediction_of_custom_data=model.predict(scaled_custom_data)
            return prediction_of_custom_data
        except Exception as e:
            raise CustomException(e,sys)
        
        
    

# This class is responsibile to fetch all the form data coming from web application   
class CustomData:
    def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,
                 test_preparation_course:str,writing_score:int,reading_score:str):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score
        
    def get_dataframe(self):
        """
        function to return the custom data as dataframe  what we get from webapp. 
        """
        try:
            custom_data_dict = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "writing_score" : [self.writing_score],
                "reading_score" : [self.reading_score]
            }
            
            
            return pd.DataFrame(custom_data_dict)
        except :
            pass
    
