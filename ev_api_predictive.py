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
    print(f"[INFO] Model Expects: {model_features}", flush=True)
except FileNotFoundError:
    print(f"[FATAL] '{MODEL_FILE}' not found.", flush=True)
    sys.exit(1)

# --- INTELLIGENT FEATURE ALIGNER ---
def align_features(input_df, target_features):
    """
    1. Maps PHP keys to Model keys.
    2. CALIBRATES Unit Mismatches (Ratio -> Magnitude).
    3. Synthesizes missing Time/History data based on signal intensity.
    """
    aligned_df = pd.DataFrame(index=input_df.index)
    
    # --- 1. CALIBRATION CONFIG (THE FIX) ---
    # Laravel sends 'voltage_instability' as a ratio (0.0 - 1.0).
    # The Model expects it as Volts (0.0 - 230.0).
    # We multiply by 230 to restore the magnitude.
    calibration_map = {
        'voltage_instability': 230.0, 
        'voltage': 1.0,               
        'avg_peak_temp': 1.0,
        'error_rate': 1.0
    }

    # --- 2. Time Features ---
    now = datetime.now()
    current_month = now.month
    current_day = now.weekday()

    # --- 3. Dictionary: PHP Name -> Model Name ---
    alias_map = {
        'avg_peak_temp': ['avg_peak_temp', 'temperature', 'temp'],
        'voltage_instability': ['voltage_instability', 'voltage', 'variance', 'voltagemeasure'],
        'error_rate': ['error_rate', 'error'],
        'sessions_today': ['sessions_today', 'sessions'],
        'ambient_temp': ['ambient_temp', 'ambient']
    }

    # --- 4. Build DataFrame ---
    for feature in target_features:
        
        # A. Time Features
        if feature == 'month_of_year':
            aligned_df[feature] = current_month
            continue
        if feature == 'day_of_week':
            aligned_df[feature] = current_day
            continue
        if 'model_type' in feature:
            aligned_df[feature] = 0.0
            continue

        # B. Direct & Alias Matching with Calibration
        found = False
        val = 0.0
        
        # Search aliases
        search_keys = [feature]
        for php_key, aliases in alias_map.items():
            if feature in aliases:
                search_keys.append(php_key)
        
        for key in search_keys:
            if key in input_df.columns:
                raw_val = float(input_df[key].iloc[0])
                
                # Apply Multiplier logic
                multiplier = 1.0
                if 'voltage' in feature or 'instability' in feature:
                     # If value is small (<5.0), it's a ratio. Boost it.
                     if raw_val < 5.0: 
                         multiplier = calibration_map['voltage_instability']
                
                val = raw_val * multiplier
                aligned_df[feature] = val
                found = True
                break
        
        if found: continue

        # C. Synthesize Historical Data (Rolling Means/Std)
        # We 'Project' the current fault into the history fields to ensure the model reacts.
        # This fixes the issue where missing history makes the model think it's safe.
        
        # 1. Handle Standard Deviation (_std)
        if 'std' in feature:
            # Find a proxy value in the currently aligned data
            proxy_val = 0
            if 'voltage' in feature:
                # Find the main voltage column
                for col in aligned_df.columns:
                    if 'voltage' in col and 'std' not in col:
                        proxy_val = aligned_df[col].iloc[0]
                        break
                # If Voltage Deviation is High (>20V), assume High Variance
                if proxy_val > 15: 
                    aligned_df[feature] = proxy_val * 0.4 
                else:
                    aligned_df[feature] = proxy_val * 0.05

            elif 'temp' in feature:
                for col in aligned_df.columns:
                    if 'temp' in col and 'std' not in col:
                        proxy_val = aligned_df[col].iloc[0]
                        break
                if proxy_val > 80:
                    aligned_df[feature] = proxy_val * 0.15
                else:
                    aligned_df[feature] = proxy_val * 0.02
            else:
                 aligned_df[feature] = 0.0
            continue

        # 2. Handle Rolling Means (_roll_mean)
        if 'roll_mean' in feature:
            root_name = feature.split('_roll')[0]
            # Use current value as the rolling mean estimate
            # This assumes "If it's bad now, it's been bad for a while" (Safety First)
            if root_name in aligned_df.columns:
                aligned_df[feature] = aligned_df[root_name]
            else:
                # Fallback search
                for col in aligned_df.columns:
                    if root_name in col:
                        aligned_df[feature] = aligned_df[col]
                        break
            continue

        # D. Final Fallback
        aligned_df[feature] = 0.0

    # 5. STRICT REORDERING
    aligned_df = aligned_df[target_features]
    return aligned_df

# --- DIAGNOSTICS ---
def get_root_cause(row_df, is_high_risk=False):
    reasons = []
    t = 0
    v = 0
    e = 0

    # Extract values safely
    for col in row_df.index:
        if 'avg_peak_temp' in col and 'mean' not in col: t = row_df[col]; break
    for col in row_df.index:
        if 'voltage' in col and 'mean' not in col: v = row_df[col]; break
    for col in row_df.index:
        if 'error' in col and 'mean' not in col: e = row_df[col]; break

    if t > 60: reasons.append(f"Overheating ({t:.1f}C)")
    # Note: v > 15 because we calibrated it to be Volts, not ratio!
    if v > 15: reasons.append(f"Voltage Instability ({v:.1f}V dev)")
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
        
        # 1. Align Features (Apply Calibration)
        df_final = align_features(df_raw, model_features)

        # 2. Predict
        probabilities = pipeline.predict_proba(df_final)[:, 1]

        results = []
        for i, prob in enumerate(probabilities):
            prob_val = float(prob)
            
            # 3. Threshold Logic
            status = "Normal"
            risk_level = "Low"
            is_risk_high = False
            
            # > 0.50 is the standard threshold for binary classification
            if prob_val > 0.50:
                status = "Need Attention"
                risk_level = "High"
                is_risk_high = True

            root_cause = get_root_cause(df_final.iloc[i], is_high_risk=is_risk_high)
            
            # LOGGING (Only log relevant events)
            if is_risk_high or prob_val > 0.3:
                raw_t = df_final.iloc[i].get('avg_peak_temp', 0)
                raw_v = df_final.iloc[i].get('voltage_instability', 0)
                print(f"🚨 [RISK DETECTED] Prob: {prob_val:.4f} | T={raw_t:.1f}, V={raw_v:.1f} | {root_cause}", flush=True)

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
        print(f"[ERROR] {e}", flush=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Diagnostic Server on Port 5000...", flush=True)
    serve(app, host='0.0.0.0', port=5000)