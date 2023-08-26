from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictionPipeline
from src.exception import CustomException

application = Flask(__name__)

app=application

#Route

@app.route('/')
def index():
    return render_template("index.html")
        
        
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    try: 
        if request.method=='GET':
            return render_template("home.html")
        else:
            data = CustomData(
                gender = request.form.get('gender'),
                race_ethnicity= request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch = request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                writing_score=request.form.get('writing_score'),
                reading_score=request.form.get('reading_score')
                
            )
            custom_data_df=data.get_dataframe()
            print(custom_data_df)
            
            #Do the prediction using custom data of dataframe format
            prediction_pipeline = PredictionPipeline()
            prediction_results_custom=prediction_pipeline.predict(custom_data_df)
            
            return render_template('home.html',prediction_results_custom=prediction_results_custom[0])
    except Exception as e:
        raise CustomException(e,sys)
    
    
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)  
        
        
