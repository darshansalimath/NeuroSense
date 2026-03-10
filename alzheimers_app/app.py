import os
import warnings
warnings.filterwarnings("ignore")  # suppress other warnings
from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objs as go
import requests
import json
import os.path

# Suppress TensorFlow warnings/info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # optional, disable oneDNN if needed

app = Flask(__name__)
# app.secret_key = os.environ.get("FLASK_SECRET", "supersecretkey")

# Paths to model, scaler, and geocoding cache
MODEL_PATH = "alzheimers_model.h5"
SCALER_PATH = "scaler.pkl"
GEOCODE_CACHE_PATH = "geocode_cache.json"

# Load model & scaler
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Ensure alzheimers_model.h5 and scaler.pkl are present in the app folder.")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4']

# Dummy credentials
VALID_USER = "user123"
VALID_PASS = "pw123"

# Class weights (from training)
CLASS_WEIGHTS = {0:1.259,1:0.829}

# Helpers
def login_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper

def build_plotly_json(values, channels):
    """Return Plotly figure JSON for EEG input signals."""
    x = list(range(30))  # simulate time points
    fig = go.Figure()
    for i, (ch, val) in enumerate(zip(channels, values)):
        y = [val + 0.02*np.sin(0.5*t + i) for t in x]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=ch))
    fig.update_layout(
        title="EEG Feature Preview (simulated waveform)",
        xaxis_title="Time (simulated)",
        yaxis_title="Signal Value",
        height=400
    )
    return fig.to_dict()  # convert to JSON serializable dict

def get_feature_explanations():
    return {
        "Fp1": "Left frontopolar electrode — attention/executive.",
        "Fp2": "Right frontopolar electrode — attention/executive.",
        "F7": "Left frontal — language & memory.",
        "F3": "Left frontal central — executive function.",
        "Fz": "Frontal midline — working memory/attention.",
        "F4": "Right frontal central — executive function.",
        "F8": "Right frontal — behavior/cognition.",
        "T3": "Left temporal — memory encoding.",
        "C3": "Left central — motor/sensorimotor.",
        "Cz": "Vertex central — midline reference.",
        "C4": "Right central.",
        "T4": "Right temporal — memory retrieval.",
        "T5": "Left parietal/temporal area.",
        "P3": "Left parietal — spatial awareness.",
        "Pz": "Midline parietal — attention/integration.",
        "P4": "Right parietal — spatial processing."
    }

def geocode_address(address):
    """Geocode address using Nominatim API with caching."""
    # Load cache if exists
    cache = {}
    if os.path.exists(GEOCODE_CACHE_PATH):
        with open(GEOCODE_CACHE_PATH, 'r') as f:
            cache = json.load(f)

    # Check cache first
    if address in cache:
        return cache[address]

    # Geocode using Nominatim
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": "AlzheimersApp/1.0"}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data:
            result = {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"])
            }
            # Save to cache
            cache[address] = result
            with open(GEOCODE_CACHE_PATH, 'w') as f:
                json.dump(cache, f)
            return result
        return None
    except Exception as e:
        app.logger.error(f"Geocoding error for {address}: {str(e)}")
        return None

# ----------------------
# Routes

@app.route("/", methods=["GET","POST"])
def login():
    if request.method=="POST":
        u = request.form.get("username")
        p = request.form.get("password")
        if u==VALID_USER and p==VALID_PASS:
            session["user"] = u
            return redirect(url_for("predict"))
        flash("Invalid credentials. Use user123 / pw123", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    prediction = None
    confidence = None
    plot_json = None
    inputs = [0.0]*len(FEATURES)
    explanation = get_feature_explanations()  # always define

    if request.method=="POST":
        try:
            # Read inputs
            inputs = [float(request.form.get(f,0.0)) for f in FEATURES]

            # Scale & predict
            x = np.array(inputs).reshape(1,-1)
            x_scaled = scaler.transform(x)
            prob = float(model.predict(x_scaled)[0][0])
            prediction = 1 if prob>0.5 else 0
            confidence = round(prob*100,2)

            # Build Plotly visualization
            plot_json = build_plotly_json(inputs, FEATURES)

        except Exception as e:
            flash(f"Error processing input: {str(e)}", "danger")
            return redirect(url_for("predict"))

    return render_template(
        "predict.html",
        features=FEATURES,
        explanation=explanation,
        inputs=inputs,
        prediction=prediction,
        confidence=confidence,
        plot_json=plot_json,
        class_weights=[CLASS_WEIGHTS[0], CLASS_WEIGHTS[1]]
    )

# Load hospital data
try:
    hospitals_df = pd.read_csv("alz_hospitals.csv")
    # Compute unique cities for dropdown (sorted alphabetically)
    unique_cities = sorted(hospitals_df['city'].unique())
except FileNotFoundError:
    raise FileNotFoundError("alz_hospitals.csv not found in the app directory.")

@app.route("/hospitals", methods=["GET", "POST"])
@login_required
def hospitals():
    try:
        # Ensure city column is string
        hospitals_df["city"] = hospitals_df["city"].astype(str)
        city = request.form.get("city", "").strip()
        sort_by = request.form.get("sort_by", "name_asc")

        filtered = hospitals_df.copy()
        if city:
            filtered = filtered[filtered["city"].str.contains(city, case=False, na=False)]
            if filtered.empty:
                flash(f"No hospitals found for city: {city}", "info")

        # Sort the filtered DataFrame
        if sort_by == "name_asc":
            filtered = filtered.sort_values(by="name", ascending=True)
        elif sort_by == "name_desc":
            filtered = filtered.sort_values(by="name", ascending=False)
        elif sort_by == "city_asc":
            filtered = filtered.sort_values(by="city", ascending=True)
        elif sort_by == "city_desc":
            filtered = filtered.sort_values(by="city", ascending=False)

        # Geocode hospital addresses
        hospital_list = filtered.to_dict(orient="records")
        for hospital in hospital_list:
            coords = geocode_address(hospital["address"])
            hospital["lat"] = coords["lat"] if coords else None
            hospital["lon"] = coords["lon"] if coords else None

        return render_template(
            "hospitals.html",
            hospitals=hospital_list,
            city=city,
            unique_cities=unique_cities,
            sort_by=sort_by,
            hospital_count=len(hospital_list)
        )

    except Exception as e:
        flash(f"Error loading hospitals: {str(e)}", "danger")
        return render_template(
            "hospitals.html",
            hospitals=[],
            city=city,
            unique_cities=unique_cities,
            sort_by="name_asc",
            hospital_count=0
        )

if __name__ == "__main__":
    app.run(debug=True)