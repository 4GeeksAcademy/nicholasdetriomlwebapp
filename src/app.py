from flask import Flask, request, render_template
from joblib import load
from pandas import DataFrame

app = Flask(__name__)
pipeline = load('trained_model.joblib')

def preprocess_input(val1, val2, val3, val4, val5, val6):
    # Mapping string labels to numerical values
    val1_mapping = {"Urban": 1, "Suburban": 2, "Rural": 3}
    val2_mapping = {"Public": 1, "Non-public": 2}
    val3_mapping = {"Standard": 1, "Experimental": 2}
    val5_mapping = {"Female": 0, "Male": 1}
    val6_mapping = {"Does not qualify": 0, "Qualifies for reduced/free lunch": 1}

    # Apply mapping
    val1 = val1_mapping.get(val1, val1)  # If not found, use original value
    val2 = val2_mapping.get(val2, val2)
    val3 = val3_mapping.get(val3, val3)
    val5 = val5_mapping.get(val5, val5)
    val6 = val6_mapping.get(val6, val6)

    # Convert val4 to float
    val4 = float(val4)

    return {"school_setting": val1, "school_type": val2, "teaching_method": val3, "n_student": val4, "gender": val5, "lunch": val6}

@app.route("/", methods=["GET", "POST"])

def index():
    if request.method == "POST":
        
        # Obtain values from form
        val1 = request.form["val1"]
        val2 = request.form["val2"]
        val3 = request.form["val3"]
        val4 = float(request.form["val4"])
        val5 = request.form["val5"]
        val6 = request.form["val6"]

        # Preprocess the input data
        data_processed = preprocess_input(val1, val2, val3, val4, val5, val6)

        # Convert the preprocessed data to a DataFrame
        data_processed_df = DataFrame([data_processed])

        # Predict using the pipeline
        prediction = str(pipeline.predict(data_processed_df)[0])
        pred_class = prediction
    else:
        pred_class = None
    
    return render_template("index.html", prediction=pred_class)