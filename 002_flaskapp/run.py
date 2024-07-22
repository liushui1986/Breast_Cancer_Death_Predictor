from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Default values
DEFAULT_AGE = 50
DEFAULT_YEAR = 60
DEFAULT_NUM_AXI = 3

@app.route('/', methods=["GET", "POST"])
def index():
    # Set default values
    age = DEFAULT_AGE
    year = DEFAULT_YEAR
    num_Axi = DEFAULT_NUM_AXI
    pred = ""

    if request.method == "POST":
    # Get values from form or use default values
        age = request.form.get("age", DEFAULT_AGE)
        year = request.form.get("year", DEFAULT_YEAR)
        num_Axi = request.form.get("num_Axi", DEFAULT_NUM_AXI)

    # Convert to float and create prediction input
        X = np.array([[float(age), float(year), float(num_Axi)]])
        pred = model.predict_proba(X)[0][1]
    
    return render_template("index.html", age=age, year=year, num_Axi=num_Axi, pred=pred)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
