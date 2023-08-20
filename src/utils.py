import numpy as np
import pandas as pd
import os
import sys
import dill

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_squared_error,r2_score

def save_object(file_path,obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def get_metrics(actual_value,preidct_value):
    mse = mean_squared_error(actual_value,preidct_value)   
    rmse = np.sqrt(mse)
    r2_score_value = r2_score(actual_value,preidct_value)
    return (mse,rmse,r2_score_value)

    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        model_report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            #Fit the model to train data
            model.fit(X_train,y_train)
            
            y_train_predict = model.predict(X_train)
            y_test_predict =  model.predict(X_test)
            
            #Calculate metrics
            model_train_mse,model_train_rmse,model_train_r2_score = get_metrics(y_train,y_train_predict)
            model_test_mse,model_test_rmse,model_test_r2_score  = get_metrics(y_test,y_test_predict)
            
            model_report[list(models.keys())[i]] = model_test_r2_score
            
        return model_report
            
    except Exception as e:
        raise CustomException(e,sys)