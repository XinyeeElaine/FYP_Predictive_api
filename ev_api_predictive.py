import pandas as pd
import joblib
import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from waitress import serve

# --- CONFIGURATION ---
MODEL_FILE = 'ev_charger_model.pkl'
app = Flask(__name__)

# --- LOAD MODEL ---
pipeline = None
model_features = []

try:
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    data_pkg = joblib.load(model_path)
    pipeline = data_pkg['pipeline']
    model_features = data_pkg['features']
    print(f"✅ SUCCESS: Loaded Golden Model with {len(model_features)} features.")
except FileNotFoundError:
    print(f"❌ FATAL: '{MODEL_FILE}' not found.")
    sys.exit(1)

# --- DIAGNOSTIC HELPER FUNCTIONS ---
def get_root_cause(row_df, pipeline, feature_names, is_high_risk=False):
    try:
        scaler = pipeline.named_steps['scaler']
        means = scaler.mean_
        scales = np.where(scaler.scale_ == 0, 1, scaler.scale_)
        raw_values = row_df[feature_names].values.flatten()
        z_scores = (raw_values - means) / scales
        
        contributions = []
        for name, z in zip(feature_names, z_scores):
            # We care about positive deviations (higher than normal)
            if z > 0: contributions.append((name, z))
        
        contributions.sort(key=lambda x: x[1], reverse=True)
        top_drivers = contributions[:3]
        
        # --- UPDATE: SMART THRESHOLD ---
        # If the risk is High, we MUST find a reason, so we lower the threshold to 0.5.
        # If risk is Low, we keep it strict (1.5) to avoid false alarms.
        threshold = 0.5 if is_high_risk else 1.5

        if not top_drivers or top_drivers[0][1] < threshold:
            return "Normal Range"
        
        explanation = []
        for feat, score in top_drivers:
            # Clean up names for the user
            readable = feat.replace('_roll_mean_14d', '').replace('_roll_mean_7d', '')\
                           .replace('_roll_std_14d', ' Var').replace('_', ' ').title()
            explanation.append(f"{readable} ({score:.1f}x)")
            
        return ", ".join(explanation)
    except Exception:
        return "Diagnostics Unavailable"

def categorize_failure(text):
    # This creates the "Problem Found" text (e.g., OVERHEATING)
    if "Temp" in text: return "OVERHEATING"
    if "Volt" in text: return "POWER SURGE"
    if "Error" in text: return "SOFTWARE CRASH"
    
    # If we have a generic variance but no specific keyword matches
    if "Var" in text: return "SIGNAL INSTABILITY"
    
    return "PERFORMANCE DEGRADATION"

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, list):
            data = [data]

        df = pd.DataFrame(data)

        # Fill missing columns
        for col in model_features:
            if col not in df.columns: df[col] = 0.0
        df_final = df[model_features]

        # 1. Predict Probability
        probabilities = pipeline.predict_proba(df_final)[:, 1]

        results = []
        for i, prob in enumerate(probabilities):
            prob_val = float(prob)
            
            # 2. Determine Status FIRST (so we know if we need to force a diagnosis)
            # Note: I see you used 0.60 in your code.
            if prob_val > 0.60:
                status = "Need Attention"
                risk_level = "High" 
                is_risk_high = True
            else:
                status = "Normal"
                risk_level = "Low"
                is_risk_high = False

            # 3. Run Diagnostics with the new 'is_high_risk' flag
            row_data = df_final.iloc[[i]]
            root_cause = get_root_cause(row_data, pipeline, model_features, is_high_risk=is_risk_high)
            
            # 4. Categorize
            if is_risk_high:
                category = categorize_failure(root_cause)
            else:
                category = "-"

            results.append({
                'status': status,
                'risk_level': risk_level,
                'probability': prob_val,
                'failure_category': category,
                'root_cause': root_cause
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Diagnostic Server on Port 5000...")
    serve(app, host='0.0.0.0', port=5000)