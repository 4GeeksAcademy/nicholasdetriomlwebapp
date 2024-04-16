from flask import Flask, request, render_template
from joblib import load
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pandas import DataFrame
import numpy as np
import pandas as pd

app = Flask(__name__)
pipeline = load('trained_model3.joblib')


def preprocess_input(val1, val2, val3, val4, val5):
    # Create a dictionary from the input values
    input_dict = {
    "school_setting": [val1],
    "school_type": [val2],
    "teaching_method": [val3],
    "gender": [val4],
    "lunch": [val5]
    }
    
    # Convert dictionary to DataFrame
    input_df = DataFrame(input_dict)
    

    
    return input_df

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        # Obtain values from form
        val1 = request.form["val1"]
        val2 = request.form["val2"]
        val3 = request.form["val3"]
        val4 = request.form["val4"]
        val5 = request.form["val5"]

        # Preprocess the input data
        data_processed = preprocess_input(val1, val2, val3, val4, val5)
        
        # Predict using the pipeline
        prediction = pipeline.predict(data_processed)[0]
        
        # Convert prediction to string
        pred_class = str(prediction)
    else:
        pred_class = None
    
    return render_template("index.html", prediction=pred_class)