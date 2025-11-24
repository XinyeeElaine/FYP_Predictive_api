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
    print(f"[SUCCESS] Loaded Model with {len(model_features)} features.")
    print(f"[INFO] MODEL EXPECTS: {model_features}") 
except FileNotFoundError:
    print(f"[FATAL] '{MODEL_FILE}' not found.")
    sys.exit(1)

# --- INTELLIGENT FEATURE ALIGNER ---
def align_features(input_df, target_features):
    """
    Maps incoming PHP keys to Model keys, generates time data, 
    and synthesizes missing history.
    """
    aligned_df = pd.DataFrame(index=input_df.index)
    
    # 1. Generate Time Features (Missing in Payload)
    # We generate these here because the PHP controller typically sends raw timestamps
    # but the model expects explicit integer columns (e.g. 1-12, 0-6).
    now = datetime.now()
    if 'month_of_year' in target_features:
        aligned_df['month_of_year'] = now.month
    if 'day_of_week' in target_features:
        aligned_df['day_of_week'] = now.weekday() # 0=Monday

    # 2. Map Known PHP Keys -> Model Keys (Base Values)
    # We only map to "Base" or "Mean" values here. We do NOT map to Std Dev yet.
    alias_map = {
        'avg_peak_temp': ['avg_peak_temp', 'temperature', 'temp'],
        'voltage_instability': ['voltage_instability', 'voltage', 'variance'],
        'error_rate': ['error_rate', 'error'],
        'sessions_today': ['sessions_today', 'sessions'],
        'ambient_temp': ['ambient_temp', 'ambient']
    }

    # 3. Iterate through TARGET features
    for feature in target_features:
        # Skip if we already filled it (like Time features)
        if feature in aligned_df.columns: continue

        # A. Exact Match
        if feature in input_df.columns:
            aligned_df[feature] = input_df[feature]
            continue
            
        # B. Synonym Match (Strict)
        found = False
        for php_key, aliases in alias_map.items():
            if php_key in input_df.columns:
                if feature in aliases:
                    aligned_df[feature] = input_df[php_key]
                    found = True
                    break
                
                # C. Smart Rolling Mean Mapping
                # Only map raw values to _mean columns. NOT _std columns.
                if 'mean' in feature and php_key in feature:
                     aligned_df[feature] = input_df[php_key]
                     found = True
                     break
        
        if found: continue

        # D. Synthesize Standard Deviation (The "Stability" Fix)
        # If model needs 'std', we derive it from the 'mean' we just populated.
        if 'std' in feature:
            # Find the base name (e.g. 'avg_peak_temp_roll_std_14d' -> 'avg_peak_temp_roll_mean_14d')
            base_mean = feature.replace('_std_', '_mean_')
            if base_mean in aligned_df.columns:
                # Simulate small variance (5% of the value)
                aligned_df[feature] = aligned_df[base_mean] * 0.05
                continue
            
            # Fallback: Try to find raw base name
            root_name = feature.split('_roll')[0] # e.g. 'avg_peak_temp'
            if root_name in input_df.columns:
                 aligned_df[feature] = input_df[root_name] * 0.05
                 continue

        # E. Handle Categoricals (Model Types)
        if 'model_type' in feature:
            aligned_df[feature] = 0.0 # Default to 0 (Generic Model)
            continue

        # F. Fallback
        aligned_df[feature] = 0.0

    return aligned_df

# --- DIAGNOSTIC HELPERS ---
def get_root_cause(row_df, is_high_risk=False):
    reasons = []
    
    # Safe getter
    def val(col): return row_df.get(col, 0)

    # Check raw inputs from the ALIGNED dataframe
    temp = val('avg_peak_temp')
    volt = val('voltage_instability')
    err  = val('error_rate')

    if temp > 40: reasons.append(f"Overheating ({temp:.1f}C)")
    if volt > 0.1: reasons.append(f"Voltage Instability ({volt:.2f})")
    if err > 0.1: reasons.append(f"High Error Rate ({err:.2f})")
    
    if not reasons and is_high_risk:
        return "Anomaly Detected (Pattern)"
        
    return ", ".join(reasons) if reasons else "Normal Range"

def categorize_failure(text):
    text = str(text).upper()
    if "OVERHEATING" in text or "TEMP" in text: return "OVERHEATING"
    if "VOLTAGE" in text or "INSTABILITY" in text: return "POWER QUALITY"
    if "ERROR" in text: return "SOFTWARE ERROR"
    return "PREDICTIVE ALERT"

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, list): data = [data]
        
        df_raw = pd.DataFrame(data)
        
        # 1. Align Features
        df_final = align_features(df_raw, model_features)

        # 2. Predict
        probabilities = pipeline.predict_proba(df_final)[:, 1]

        results = []
        for i, prob in enumerate(probabilities):
            prob_val = float(prob)
            
            # --- SAFETY OVERRIDE ---
            # Retrieve RAW values for sanity check
            raw_temp = df_final.iloc[i].get('avg_peak_temp', 0)
            raw_err = df_final.iloc[i].get('error_rate', 0)
            
            # Force high risk if physical limits exceeded
            if raw_temp > 80: prob_val = max(prob_val, 0.98)
            if raw_err > 0.2: prob_val = max(prob_val, 0.95)

            is_high = prob_val > 0.55
            
            # Root Cause Analysis
            root_cause = get_root_cause(df_final.iloc[i], is_high)

            results.append({
                'status': "Need Attention" if is_high else "Normal",
                'risk_level': "High" if is_high else "Low",
                'probability': round(prob_val, 4),
                'root_cause': root_cause,
                'failure_category': categorize_failure(root_cause) if is_high else "-"
            })

        return jsonify(results)

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Intelligent Diagnostic Server on Port 5000...")
    serve(app, host='0.0.0.0', port=5000)