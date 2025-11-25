import pandas as pd
import joblib
import os
import sys
import numpy as np
from datetime import datetime
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
    print(f"[SUCCESS] Loaded Model with {len(model_features)} features.", flush=True)
except FileNotFoundError:
    print(f"[FATAL] '{MODEL_FILE}' not found.", flush=True)
    sys.exit(1)

# --- SIMPLE FEATURE ALIGNER (No Calibration Needed) ---
def align_features(input_df, target_features):
    """
    1. Maps PHP keys to Model keys.
    2. Copies current values to history if history is missing.
    3. NO MATHEMATICAL SCALING (Model now natively understands small numbers).
    """
    aligned_df = pd.DataFrame(index=input_df.index)
    
    # Mapping Dictionary
    alias_map = {
        'avg_peak_temp': ['avg_peak_temp', 'temperature', 'temp'],
        'voltage_instability': ['voltage_instability', 'voltage', 'variance', 'voltagemeasure'],
        'error_rate': ['error_rate', 'error'],
        'sessions_today': ['sessions_today', 'sessions'],
        'ambient_temp': ['ambient_temp', 'ambient']
    }

    # Time Features
    now = datetime.now()
    
    for feature in target_features:
        # A. Time Features
        if feature == 'month_of_year': aligned_df[feature] = now.month; continue
        if feature == 'day_of_week': aligned_df[feature] = now.weekday(); continue
        if 'model_type' in feature: aligned_df[feature] = 0.0; continue # Default to 0 if missing

        # B. Direct Match or Alias
        # We try to find the data in the input
        if feature in input_df.columns:
            aligned_df[feature] = input_df[feature]
            continue
            
        found_alias = False
        for php_key, aliases in alias_map.items():
            if feature in aliases and php_key in input_df.columns:
                aligned_df[feature] = input_df[php_key]
                found_alias = True
                break
        
        if found_alias: continue

        # C. Handle Missing History (Copy Current to History)
        # If Laravel sends 'voltage' but not 'voltage_roll_mean', use 'voltage' as the mean.
        if 'roll_mean' in feature:
            root = feature.split('_roll')[0]
            if root in aligned_df.columns:
                aligned_df[feature] = aligned_df[root]
                continue
            # Fallback check for aliases
            for php_key, aliases in alias_map.items():
                if root in aliases and php_key in input_df.columns:
                     aligned_df[feature] = input_df[php_key]
                     break
            continue

        # D. Handle Missing Std Dev
        if 'roll_std' in feature:
            # We assume a small default noise if missing
            aligned_df[feature] = 0.05
            continue

        # E. Default
        aligned_df[feature] = 0.0

    return aligned_df[target_features]

# --- DIAGNOSTICS ---
def get_root_cause(row_df, is_high_risk=False):
    reasons = []
    
    # Safely get values
    t = row_df.get('avg_peak_temp', 0)
    v = row_df.get('voltage_instability', 0)
    e = row_df.get('error_rate', 0)

    # Thresholds match your NEW Training Data
    if t > 80: reasons.append(f"Overheating ({t:.1f}C)")
    if v > 0.08: reasons.append(f"Grid Instability (Idx: {v:.2f})")
    if e > 0.5: reasons.append(f"Software Errors")
    
    if not reasons and is_high_risk:
        return "Anomaly Detected (Pattern)"
        
    return ", ".join(reasons) if reasons else "Normal Range"

def categorize_failure(text):
    text = str(text).upper()
    if "OVERHEATING" in text: return "OVERHEATING"
    if "GRID" in text or "VOLT" in text: return "POWER QUALITY"
    if "SOFTWARE" in text or "ERROR" in text: return "SOFTWARE ERROR"
    return "PREDICTIVE ALERT"

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, list): data = [data]
        df_raw = pd.DataFrame(data)
        
        # 1. Align (Raw values pass through - NO x230 MULTIPLIER)
        df_final = align_features(df_raw, model_features)
        
        # 2. Predict
        probabilities = pipeline.predict_proba(df_final)[:, 1]
        
        results = []
        for i, prob in enumerate(probabilities):
            prob_val = float(prob)
            
            # 3. Determine Status (Threshold 0.50)
            status = "Normal"
            risk = "Low"
            is_high = False
            
            if prob_val > 0.50:
                status = "Need Attention"
                risk = "High"
                is_high = True
            
            # 4. Diagnostics
            root_cause = get_root_cause(df_final.iloc[i], is_high_risk=is_high)
            category = categorize_failure(root_cause) if is_high else "-"
            
            # Log high risks
            if is_high:
                 print(f"[RISK] Prob:{prob_val:.4f} | {root_cause}", flush=True)

            results.append({
                'status': status,
                'risk_level': risk,
                'probability': round(prob_val, 4),
                'failure_category': category,
                'root_cause': root_cause
            })

        return jsonify(results)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Clean AI Server...", flush=True)
    serve(app, host='0.0.0.0', port=5000)