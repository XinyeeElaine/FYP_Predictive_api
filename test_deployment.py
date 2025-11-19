import requests
import json
import pandas as pd

# --- CONFIGURATION ---
url = 'http://127.0.0.1:5000/predict'

# --- HELPER: PHYSICS ENGINE ---
# This simulates the variance (jitter) found in real sensors.
# If a reading is Extreme, the Variance is likely High (5.0).
# If a reading is Normal, the Variance is Low (1.0).
def enrich_scenario(sc):
    enriched = sc.copy()
    checks = [
        ('error_rate', 5.0), 
        ('voltage_instability', 0.2), 
        ('avg_peak_temp', 80.0)
    ]
    for base, threshold in checks:
        if base in enriched:
            val = enriched[base]
            # If value > threshold, set High Variance (5.0), else Low (1.0)
            variance = 5.0 if val > threshold else 1.0
            enriched[f'{base}_roll_std_7d'] = variance
            enriched[f'{base}_roll_std_14d'] = variance
    return enriched

# --- THE 10 SCENARIOS ---
raw_scenarios = [
    # 1. HEALTHY
    {
        "desc": "1. Perfect Health",
        "ambient_temp": 20.0, "avg_peak_temp": 35.0, 
        "voltage_instability": 0.01, "error_rate": 0.0,
        "sessions_today": 12.0,
        "avg_peak_temp_roll_mean_14d": 35.0,
        "voltage_instability_roll_mean_14d": 0.01,
        "error_rate_roll_mean_14d": 0.0
    },

    # 2. FALSE ALARM (Heatwave)
    # It's hot outside (42C), so charger is hot (95C).
    # But Voltage/Errors are fine. Should be NORMAL.
    {
        "desc": "2. Heatwave (External Heat Only)",
        "ambient_temp": 42.0, "avg_peak_temp": 95.0, 
        "voltage_instability": 0.02, "error_rate": 0.0,
        "sessions_today": 10.0,
        "avg_peak_temp_roll_mean_14d": 92.0,
        "voltage_instability_roll_mean_14d": 0.02,
        "error_rate_roll_mean_14d": 0.0
    },

    # 3. FALSE ALARM (Busy Day)
    # High usage causing slight temp rise and errors. 
    # Should be NORMAL.
    {
        "desc": "3. High Usage Stress",
        "ambient_temp": 25.0, "avg_peak_temp": 65.0, 
        "voltage_instability": 0.05, "error_rate": 5.0,
        "sessions_today": 25.0, # <--- Very Busy
        "avg_peak_temp_roll_mean_14d": 60.0,
        "voltage_instability_roll_mean_14d": 0.05,
        "error_rate_roll_mean_14d": 2.0
    },

    # 4. FALSE ALARM (Grid Issue)
    # Voltage is bad, but charger is cool/functional.
    # Should be NORMAL (Grid's fault, not charger).
    {
        "desc": "4. Dirty Grid (Bad Voltage Only)",
        "ambient_temp": 20.0, "avg_peak_temp": 40.0, 
        "voltage_instability": 0.8, # <--- Bad Input
        "error_rate": 0.0,
        "sessions_today": 10.0,
        "avg_peak_temp_roll_mean_14d": 40.0,
        "voltage_instability_roll_mean_14d": 0.75,
        "error_rate_roll_mean_14d": 0.0
    },

    # 5. WARNING (Software Loop)
    # Massive errors, but hardware is fine.
    # Should be WARNING or FAILURE (depending on threshold).
    {
        "desc": "5. Software Crash Loop",
        "ambient_temp": 20.0, "avg_peak_temp": 35.0, 
        "voltage_instability": 0.02, 
        "error_rate": 80.0, # <--- Software is broken
        "sessions_today": 8.0,
        "avg_peak_temp_roll_mean_14d": 35.0,
        "voltage_instability_roll_mean_14d": 0.02,
        "error_rate_roll_mean_14d": 75.0
    },

    # 6. FAILURE (Cooling System Dead)
    # Temp is high (98C) despite normal ambient (25C).
    # Should be WARNING/FAILURE.
    {
        "desc": "6. Cooling System Failure",
        "ambient_temp": 25.0, "avg_peak_temp": 98.0, # <--- Overheating
        "voltage_instability": 0.05, 
        "error_rate": 5.0,
        "sessions_today": 5.0,
        "avg_peak_temp_roll_mean_14d": 95.0,
        "voltage_instability_roll_mean_14d": 0.05,
        "error_rate_roll_mean_14d": 5.0
    },

    # 7. FAILURE (Power Supply Fried)
    # Voltage is dangerously high (1.5). 
    # Should be CRITICAL.
    {
        "desc": "7. Power Supply Explosion",
        "ambient_temp": 20.0, "avg_peak_temp": 50.0, 
        "voltage_instability": 1.5, # <--- Dangerous
        "error_rate": 20.0,
        "sessions_today": 2.0,
        "avg_peak_temp_roll_mean_14d": 45.0,
        "voltage_instability_roll_mean_14d": 1.4,
        "error_rate_roll_mean_14d": 15.0
    },

    # 8. SILENT FAILURE (Dead Controller)
    # Cold, but Voltage/Errors are bad. Usage is 1.
    # Should be CRITICAL.
    {
        "desc": "8. Controller Fried (Silent)",
        "ambient_temp": 20.0, "avg_peak_temp": 30.0, 
        "voltage_instability": 0.9, 
        "error_rate": 45.0,
        "sessions_today": 1.0, # <--- Abandoned
        "avg_peak_temp_roll_mean_14d": 30.0,
        "voltage_instability_roll_mean_14d": 0.8,
        "error_rate_roll_mean_14d": 40.0
    },

    # 9. DEGRADATION (Drifting Sensors)
    # Nothing is "Critical", but everything is "Bad".
    # Should be WARNING.
    {
        "desc": "9. General Degradation",
        "ambient_temp": 20.0, "avg_peak_temp": 65.0, 
        "voltage_instability": 0.3, 
        "error_rate": 15.0,
        "sessions_today": 6.0,
        "avg_peak_temp_roll_mean_14d": 60.0,
        "voltage_instability_roll_mean_14d": 0.25,
        "error_rate_roll_mean_14d": 12.0
    },

    # 10. CATASTROPHE
    # Everything is broken.
    # Should be CRITICAL (100%).
    {
        "desc": "10. Total Meltdown",
        "ambient_temp": 25.0, "avg_peak_temp": 105.0, 
        "voltage_instability": 1.2, 
        "error_rate": 60.0,
        "sessions_today": 0.0, 
        "avg_peak_temp_roll_mean_14d": 100.0,
        "voltage_instability_roll_mean_14d": 1.1,
        "error_rate_roll_mean_14d": 55.0
    }
]

# --- EXECUTE ---
payload = [enrich_scenario(sc) for sc in raw_scenarios]

try:
    print(f"Sending {len(payload)} scenarios to API...\n")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        results = response.json()
        
        # Pretty Print Table Header
        print(f"{'SCENARIO':<35} | {'STATUS':<10} | {'ROOT CAUSE (Diagnostics)'}")
        print("-" * 100)
        
        for i, res in enumerate(results):
            desc = raw_scenarios[i]['desc']
            status = res['status']
            # If there is a root cause, grab it, otherwise show "-"
            cause = res.get('root_cause', '-')
            
            # Truncate cause if it's too long for the table
            if len(cause) > 50: cause = cause[:47] + "..."
            
            print(f"{desc:<35} | {status:<10} | {cause}")
            
    else:
        print("API Error:", response.text)

except Exception as e:
    print("Connection Failed:", e)
    print("Make sure test_deployment.py is running!")