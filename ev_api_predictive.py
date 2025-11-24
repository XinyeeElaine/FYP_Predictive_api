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
    1. Maps PHP keys to Model keys.
    2. Generates missing Time data.
    3. Synthesizes missing History data (Preprocessing only).
    4. ENFORCES STRICT COLUMN ORDER.
    """
    aligned_df = pd.DataFrame(index=input_df.index)
    
    # 1. Generate Time Features (Critical for Model)
    now = datetime.now()
    current_month = now.month
    current_day = now.weekday()

    # 2. Dictionary: PHP Name -> Potential Model Name Match
    alias_map = {
        'avg_peak_temp': ['avg_peak_temp', 'temperature', 'temp'],
        'voltage_instability': ['voltage_instability', 'voltage', 'variance'],
        'error_rate': ['error_rate', 'error'],
        'sessions_today': ['sessions_today', 'sessions'],
        'ambient_temp': ['ambient_temp', 'ambient']
    }

    # 3. Build the DataFrame column by column based on what the MODEL needs
    for feature in target_features:
        
        # --- A. Time Features ---
        if feature == 'month_of_year':
            aligned_df[feature] = current_month
            continue
        if feature == 'day_of_week':
            aligned_df[feature] = current_day
            continue
        if 'model_type' in feature:
            aligned_df[feature] = 0.0
            continue

        # --- B. Direct Match ---
        if feature in input_df.columns:
            aligned_df[feature] = input_df[feature]
            continue
            
        # --- C. Synonym Match ---
        found = False
        for php_key, aliases in alias_map.items():
            if php_key in input_df.columns:
                # 1. Exact alias match
                if feature in aliases:
                    aligned_df[feature] = input_df[php_key]
                    found = True
                    break
                
                # 2. Rolling Mean Filling (Use raw value to fill missing history)
                if 'mean' in feature and php_key in feature:
                     aligned_df[feature] = input_df[php_key]
                     found = True
                     break
        
        if found: continue

        # --- D. Synthesize Standard Deviation ---
        # (This is PREPROCESSING, not an override. We need valid inputs for the AI.)
        if 'std' in feature:
            base_mean_name = feature.replace('_std', '_mean')
            if base_mean_name in aligned_df.columns:
                aligned_df[feature] = aligned_df[base_mean_name] * 0.05
                continue
            
            root_name = feature.split('_roll')[0]
            if root_name in input_df.columns:
                 aligned_df[feature] = input_df[root_name] * 0.05
                 continue

        # --- E. Final Fallback ---
        aligned_df[feature] = 0.0

    # 4. STRICT REORDERING (Prevents "Feature names must match" error)
    aligned_df = aligned_df[target_features]
    
    return aligned_df

# --- DIAGNOSTICS ---
def get_root_cause(row_df, is_high_risk=False):
    reasons = []
    
    # We inspect the ALIGNED data (what the model saw)
    t = 0
    v = 0
    e = 0

    # Helper to find values in the aligned dataframe regardless of naming variations
    for col in row_df.index:
        if 'avg_peak_temp' in col and 'mean' not in col: t = row_df[col]; break
        elif 'temperature' in col and 'mean' not in col: t = row_df[col]; break
    
    for col in row_df.index:
        if 'voltage' in col and 'mean' not in col: v = row_df[col]; break
    
    for col in row_df.index:
        if 'error' in col and 'mean' not in col: e = row_df[col]; break

    if t > 60: reasons.append(f"Overheating ({t:.1f}C)")
    if v > 0.15: reasons.append(f"Voltage Instability ({v:.2f})")
    if e > 0.1: reasons.append(f"High Error Rate ({e:.2f})")
    
    if not reasons and is_high_risk:
        return "Anomaly Detected (Pattern)"
        
    return ", ".join(reasons) if reasons else "Normal Range"

def categorize_failure(text):
    text = str(text).upper()
    if "OVERHEATING" in text: return "OVERHEATING"
    if "VOLTAGE" in text: return "POWER QUALITY"
    if "ERROR" in text: return "SOFTWARE ERROR"
    return "PREDICTIVE ALERT"

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, list): data = [data]
        
        df_raw = pd.DataFrame(data)
        
        # 1. Align & Reorder Features (Data Prep)
        df_final = align_features(df_raw, model_features)

        # 2. Predict (PURE AI - No Manual Rules)
        probabilities = pipeline.predict_proba(df_final)[:, 1]

        results = []
        for i, prob in enumerate(probabilities):
            prob_val = float(prob)
            
            # 3. Determine Status (Strictly > 0.60)
            if prob_val > 0.60:
                status = "Need Attention"
                risk_level = "High" 
                is_risk_high = True
            else:
                status = "Normal"
                risk_level = "Low"
                is_risk_high = False

            # 4. Run Diagnostics (For UI explanation only)
            root_cause = get_root_cause(df_final.iloc[i], is_high_risk=is_risk_high)
            
            # 5. Categorize
            if is_risk_high:
                category = categorize_failure(root_cause)
            else:
                category = "-"

            results.append({
                'status': status,
                'risk_level': risk_level,
                'probability': round(prob_val, 4),
                'failure_category': category,
                'root_cause': root_cause
            })

        return jsonify(results)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Diagnostic Server on Port 5000...")
    serve(app, host='0.0.0.0', port=5000)