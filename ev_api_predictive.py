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
    print(f" SUCCESS: Loaded Model with {len(model_features)} features.")
    print(f" MODEL EXPECTS: {model_features}") 
except FileNotFoundError:
    print(f" FATAL: '{MODEL_FILE}' not found.")
    sys.exit(1)

# --- INTELLIGENT FEATURE ALIGNER (The Fix) ---
def align_features(input_df, target_features):
    """
    Maps incoming PHP keys to Model keys and fills missing data intelligently.
    """
    aligned_df = pd.DataFrame(index=input_df.index)
    
    # 1. Dictionary of Synonyms (PHP Name -> Potential Model Names)
    # We map the INPUT (PHP) to the TARGET (Model)
    alias_map = {
        'avg_peak_temp': ['temperature', 'temp', 'peak_temp', 'avg_temp'],
        'voltage_instability': ['voltage', 'volt', 'variance', 'instability'],
        'error_rate': ['error', 'errors', 'fail_rate'],
        'sessions_today': ['sessions', 'utilization', 'usage'],
        
        # Rolling features
        'avg_peak_temp_roll_mean_14d': ['temperature_roll_mean', 'temp_roll_mean', 'avg_peak_temp_roll_mean'],
        'voltage_instability_roll_mean_14d': ['voltage_roll_mean', 'volt_roll_mean'],
    }

    # 2. Iterate through what the MODEL needs
    for feature in target_features:
        
        # Case A: Exact Match
        if feature in input_df.columns:
            aligned_df[feature] = input_df[feature]
            continue
            
        # Case B: Synonym Match
        found = False
        for php_key, aliases in alias_map.items():
            if php_key in input_df.columns:
                # Check if the target feature matches one of the aliases
                if feature in aliases:
                    aligned_df[feature] = input_df[php_key]
                    found = True
                    print(f"    Mapped '{php_key}' -> '{feature}'")
                    break
                # Check if partial string match (e.g. 'temp' in 'avg_peak_temp')
                # This is aggressive matching for rolling means
                if not found and php_key in feature and 'roll' in feature:
                     aligned_df[feature] = input_df[php_key]
                     found = True
                     print(f"    Fuzzy Mapped '{php_key}' -> '{feature}'")
                     break

        if found: continue

        # Case C: Synthesize Missing History (The "Zero Fix")
        # If the model needs 'std' (Standard Deviation) but we only have 'mean',
        # we simulate 'std' as 10% of the mean to avoid 0 values.
        if 'std' in feature or 'var' in feature:
            # Find the corresponding 'mean' column in our aligned_df
            base_name = feature.replace('std', 'mean').replace('var', 'mean')
            if base_name in aligned_df.columns:
                aligned_df[feature] = aligned_df[base_name] * 0.1 # Assume 10% variance
                print(f"   🧪 Synthesized '{feature}' from '{base_name}'")
                continue

        # Case D: Fallback to 0 (Log warning)
        print(f"   ⚠️ Missing Feature: '{feature}'. Defaulting to 0.")
        aligned_df[feature] = 0.0

    return aligned_df

# --- DIAGNOSTIC HELPERS ---
def get_root_cause(row_df, is_high_risk=False):
    """Simplified Root Cause Analysis"""
    reasons = []
    
    # Extract values regardless of column name
    def get_val(keywords):
        for col in row_df.index:
            if any(k in col for k in keywords):
                return row_df[col]
        return 0.0

    val_temp = get_val(['temp', 'Temp'])
    val_volt = get_val(['volt', 'Volt', 'instability'])
    val_err = get_val(['error', 'Error'])

    if val_temp > 50: reasons.append(f"Overheating ({val_temp:.1f}°C)")
    if val_volt > 0.15: reasons.append(f"Voltage Instability ({val_volt:.2f})")
    if val_err > 0.1: reasons.append(f"High Error Rate")
    
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
        
        # 1. Raw Data
        df_raw = pd.DataFrame(data)
        print(f"\n📥 Received: {len(df_raw)} records. Cols: {list(df_raw.columns)}")

        # 2. Align Features (Mapping)
        df_final = align_features(df_raw, model_features)

        # 3. Predict
        probabilities = pipeline.predict_proba(df_final)[:, 1]

        results = []
        for i, prob in enumerate(probabilities):
            prob_val = float(prob)
            
            # --- DEBUG OVERRIDE ---
            # If auto-mapping failed to convince the model, use Physics rules
            raw_temp = df_final.iloc[i].get('avg_peak_temp', 0) 
            # Try multiple names for temp in case mapping renamed it
            if raw_temp == 0: 
                raw_temp = df_final.iloc[i].get('temperature', df_final.iloc[i].get('temp', 0))

            if raw_temp > 80: prob_val = max(prob_val, 0.95) # Force High
            if df_final.iloc[i].get('error_rate', 0) > 0.5: prob_val = max(prob_val, 0.90)

            # Status Logic
            is_high = prob_val > 0.55
            
            results.append({
                'status': "Need Attention" if is_high else "Normal",
                'risk_level': "High" if is_high else "Low",
                'probability': round(prob_val, 4),
                'root_cause': get_root_cause(df_final.iloc[i], is_high),
                'failure_category': categorize_failure(get_root_cause(df_final.iloc[i], is_high)) if is_high else "-"
            })

        return jsonify(results)

    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Intelligent Diagnostic Server on Port 5000...")
    serve(app, host='0.0.0.0', port=5000)