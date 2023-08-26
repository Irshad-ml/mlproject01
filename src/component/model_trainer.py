import os
import sys

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

#Importing Alogirthm
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

@dataclass
class ModelTrainerConfig:
    """This class is used to create the config file for ModelTrainer Component .
       Config file means define some filepath which is required as input in this Model trainer component
       Here  we define file path of model.pkl
    """
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
print(ModelTrainerConfig().trained_model_file_path)
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_file_path):
        try:
            logging.info("Split training and test array dataset")
            X_train,y_train,X_test,y_test =(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            #Define the models
            
            models = {
                "DecisionTree":DecisionTreeRegressor(),
                "RandomForest":RandomForestRegressor(),                
                "GradientBoosting":GradientBoostingRegressor(),
                "LinearRegression":LinearRegression(),
                "CatBoost":CatBoostRegressor(),
                "AdaBoost":AdaBoostRegressor(),
            }
            
            params={
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                # "XGBRegressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            
            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,
                                               y_test=y_test,models=models,params=params)
            
            print(f"modelreport from evaluation: {model_report}")
            
            #Get the best model score
            best_model_score = max(sorted(list(model_report.values())))
            print(f"Best model score: {best_model_score}")
            
            #Get best model name 
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            print(f"Best model name: {best_model_name}")
            
            #Get the model object
            best_model=models[best_model_name]
            print(f"Best model object: {best_model}")
            
            if best_model_score < 0.7 :
                raise CustomException("No best Model found")
            logging.info(f"Found best model on training and testing dataset ")
            
            #Dumping the best model in pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted_outcome=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted_outcome)
            
            return r2_square
            
            
        except Exception as e:
            raise CustomException(e,sys)