import os
import sys

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

#Importing Alogirthm
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

@dataclass
class ModelTrainerConfig:
    """This class is used to create the config file related to ModelTrainer .
       Config file means define some filepath which is required as input in this Model trainer part
       Here  we define file path of model.pkl
    """
        model_file_path = os.path.