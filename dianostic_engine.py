import numpy as np
import pandas as pd

def get_root_cause(row_df, pipeline, feature_names):
    """
    Analyzes a single row of data to find the primary drivers of failure.
    """
    # 1. Extract the Scaler from the Pipeline
    # The scaler knows what 'Normal' looks like (Mean and Scale)
    try:
        scaler = pipeline.named_steps['scaler']
        means = scaler.mean_
        scales = scaler.scale_ # This is the Standard Deviation
    except:
        return "Cannot diagnostics: Scaler not found in pipeline."

    # 2. Calculate Z-Scores for this specific charger
    # Z = (Value - Mean) / StdDev
    # This tells us how many "Sigmas" away from normal we are.
    
    # Get raw values as array
    raw_values = row_df[feature_names].values.flatten()
    
    # Avoid division by zero
    scales = np.where(scales == 0, 1, scales)
    
    z_scores = (raw_values - means) / scales
    
    # 3. Map Z-Scores to Feature Names
    contributions = []
    for name, z in zip(feature_names, z_scores):
        # We only care about POSITIVE deviations (Values that are too High)
        # Because in your physics model, High Temp/Volt/Error = Bad.
        # We ignore negative Z-scores (e.g., unusually low temp is fine).
        if z > 0:
            contributions.append((name, z))
            
    # 4. Sort by Severity (Highest Z-Score first)
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    # 5. Interpret the Top 3 Drivers
    top_drivers = contributions[:3]
    
    if not top_drivers or top_drivers[0][1] < 2.0:
        return "No specific anomaly detected (All sensors within normal range)."
    
    # Generate Human-Readable Explanation
    explanation = []
    for feat, score in top_drivers:
        severity = "Elevated"
        if score > 3: severity = "High"
        if score > 5: severity = "CRITICAL"
        
        # Clean up feature name for display
        readable_name = feat.replace('_roll_mean_14d', ' (14d Avg)')\
                            .replace('_roll_mean_7d', ' (7d Avg)')\
                            .replace('_roll_std_14d', ' (Variance)')\
                            .replace('_', ' ').title()
        
        explanation.append(f"{readable_name}: {severity} ({score:.1f}σ)")
        
    return " | ".join(explanation)

def categorize_failure(root_cause_str):
    """
    Assigns a category label based on the text description.
    """
    text = root_cause_str.lower()
    if "temp" in text:
        return "OVERHEATING"
    elif "voltage" in text:
        return "POWER INSTABILITY"
    elif "error" in text:
        return "SOFTWARE/COMMS FAILURE"
    else:
        return "UNKNOWN ANOMALY"