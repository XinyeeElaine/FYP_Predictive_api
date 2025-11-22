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
    """
    Attempts to find root cause using Z-scores.
    If that fails, uses manual calculation but KEEPS the (Scorex) format.
    """
    # 1. Try Advanced Z-Score Method
    try:
        if 'scaler' in pipeline.named_steps:
            scaler = pipeline.named_steps['scaler']
            means = scaler.mean_
            scales = np.where(scaler.scale_ == 0, 1, scaler.scale_)
            raw_values = row_df[feature_names].values.flatten()
            z_scores = (raw_values - means) / scales
            
            contributions = []
            for name, z in zip(feature_names, z_scores):
                if z > 0: contributions.append((name, z))
            
            contributions.sort(key=lambda x: x[1], reverse=True)
            top_drivers = contributions[:3]
            
            # Threshold: 0.5 if High Risk (aggressive), 1.5 if Low Risk (strict)
            threshold = 0.5 if is_high_risk else 1.5

            if top_drivers and top_drivers[0][1] >= threshold:
                explanation = []
                for feat, score in top_drivers:
                    readable = feat.replace('_roll_mean_14d', '').replace('_roll_mean_7d', '')\
                                   .replace('_roll_std_14d', ' Var').replace('_', ' ').title()
                    explanation.append(f"{readable} ({score:.1f}x)")
                return ", ".join(explanation)

    except Exception:
        # If math fails, silently pass to the fallback below
        pass

    # 2. Manual Fallback (The "Safety Net")
    # This runs ONLY if Z-scores failed or found nothing, but the Risk is High.
    if is_high_risk:
        reasons = []
        try:
            # Helper to safely get value (returns 0.0 if column missing)
            def get_val(col):
                return row_df[col].values[0] if col in row_df.columns else 0.0

            # Check Voltage (Safety Limit: 0.1)
            val_volt = get_val('voltage_instability')
            if val_volt > 0.1:
                score = val_volt / 0.1
                reasons.append(f"Voltage Instability ({score:.1f}x)")

            # Check Temp (Safety Limit: 35)
            val_temp = get_val('avg_peak_temp')
            if val_temp > 35:
                score = val_temp / 35.0
                reasons.append(f"Avg Peak Temp ({score:.1f}x)")

            # Check Error Rate (Safety Limit: 0.05)
            val_err = get_val('error_rate')
            if val_err > 0.05:
                score = val_err / 0.05
                reasons.append(f"Error Rate ({score:.1f}x)")
            
            # Check Utilization (Limit: 20 sessions)
            val_sess = get_val('sessions_today')
            if val_sess > 20:
                score = val_sess / 20.0
                reasons.append(f"High Utilization ({score:.1f}x)")

            if reasons:
                return ", ".join(reasons)
            
            return "Anomaly Detected (1.0x)"
            
        except Exception:
            return "Data Error"

    return "Normal Range"

def categorize_failure(text):
    text = text.upper()
    if "TEMP" in text: return "OVERHEATING"
    if "VOLT" in text: return "POWER SURGE"
    if "ERROR" in text or "SOFTWARE" in text: return "SOFTWARE CRASH"
    if "VAR" in text or "INSTABILITY" in text: return "SIGNAL INSTABILITY"
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
            
            # 2. Determine Status (Threshold 0.60)
            if prob_val > 0.60:
                status = "Need Attention"
                risk_level = "High" 
                is_risk_high = True
            else:
                status = "Normal"
                risk_level = "Low"
                is_risk_high = False

            # 3. Run Diagnostics
            root_cause = get_root_cause(df_final.iloc[[i]], pipeline, model_features, is_high_risk=is_risk_high)
            
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