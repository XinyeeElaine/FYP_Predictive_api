import requests
import json
import time

# --- CONFIGURATION ---
# Replace with your actual Render URL if needed
url = 'https://fyp-predictive-api.onrender.com/predict'

# --- HELPER: PHYSICS ENGINE (Adds Variance) ---
def enrich_scenario(sc):
    enriched = sc.copy()
    # Logic: If reading is extreme, Variance (Standard Deviation) is likely High
    checks = [
        ('error_rate', 5.0), 
        ('voltage_instability', 0.2), 
        ('avg_peak_temp', 80.0)
    ]
    for base, threshold in checks:
        if base in enriched:
            val = enriched[base]
            variance = 5.0 if val > threshold else 1.0
            enriched[f'{base}_roll_std_7d'] = variance
            enriched[f'{base}_roll_std_14d'] = variance
    return enriched

# --- THE 20 SCENARIOS ---
raw_scenarios = [
    # --- GROUP 1: NORMAL OPERATIONS ---
    {
        "desc": "01. Perfect Health",
        "ambient_temp": 20.0, "avg_peak_temp": 35.0, "voltage_instability": 0.01, "error_rate": 0.0,
        "sessions_today": 12.0, "avg_peak_temp_roll_mean_14d": 35.0, 
        "voltage_instability_roll_mean_14d": 0.01, "error_rate_roll_mean_14d": 0.0
    },
    {
        "desc": "02. Cold Day (Winter)",
        "ambient_temp": -5.0, "avg_peak_temp": 15.0, "voltage_instability": 0.02, "error_rate": 0.0,
        "sessions_today": 10.0, "avg_peak_temp_roll_mean_14d": 10.0, 
        "voltage_instability_roll_mean_14d": 0.02, "error_rate_roll_mean_14d": 0.0
    },
    {
        "desc": "03. Hot Day (Summer)",
        "ambient_temp": 35.0, "avg_peak_temp": 85.0, "voltage_instability": 0.03, "error_rate": 0.0,
        "sessions_today": 8.0, "avg_peak_temp_roll_mean_14d": 80.0, 
        "voltage_instability_roll_mean_14d": 0.03, "error_rate_roll_mean_14d": 0.0
    },

    # --- GROUP 2: FALSE ALARMS (Should be Normal) ---
    {
        "desc": "04. False Alarm: Heatwave",
        "ambient_temp": 42.0, "avg_peak_temp": 98.0, "voltage_instability": 0.02, "error_rate": 0.0,
        "sessions_today": 10.0, "avg_peak_temp_roll_mean_14d": 95.0, 
        "voltage_instability_roll_mean_14d": 0.02, "error_rate_roll_mean_14d": 0.0
    },
    {
        "desc": "05. False Alarm: Heavy Usage",
        "ambient_temp": 25.0, "avg_peak_temp": 70.0, "voltage_instability": 0.08, "error_rate": 5.0,
        "sessions_today": 25.0, # <--- 25 cars!
        "avg_peak_temp_roll_mean_14d": 65.0, 
        "voltage_instability_roll_mean_14d": 0.05, "error_rate_roll_mean_14d": 2.0
    },
    {
        "desc": "06. False Alarm: Grid Fluctuation",
        "ambient_temp": 20.0, "avg_peak_temp": 40.0, 
        "voltage_instability": 0.8, # <--- Bad Input Voltage
        "error_rate": 0.0, "sessions_today": 10.0, 
        "avg_peak_temp_roll_mean_14d": 40.0, 
        "voltage_instability_roll_mean_14d": 0.75, "error_rate_roll_mean_14d": 0.0
    },

    # --- GROUP 3: WARNING SIGNS (Early Detection) ---
    {
        "desc": "07. Warning: Sensor Drift",
        "ambient_temp": 20.0, "avg_peak_temp": 45.0, "voltage_instability": 0.3, "error_rate": 15.0,
        "sessions_today": 8.0, "avg_peak_temp_roll_mean_14d": 40.0, 
        "voltage_instability_roll_mean_14d": 0.25, "error_rate_roll_mean_14d": 10.0
    },
    {
        "desc": "08. Warning: Overheating (Mild)",
        "ambient_temp": 25.0, "avg_peak_temp": 90.0, "voltage_instability": 0.1, "error_rate": 5.0,
        "sessions_today": 10.0, "avg_peak_temp_roll_mean_14d": 85.0, 
        "voltage_instability_roll_mean_14d": 0.1, "error_rate_roll_mean_14d": 2.0
    },
    {
        "desc": "09. Warning: Comms Instability",
        "ambient_temp": 20.0, "avg_peak_temp": 35.0, "voltage_instability": 0.05, "error_rate": 30.0,
        "sessions_today": 10.0, "avg_peak_temp_roll_mean_14d": 35.0, 
        "voltage_instability_roll_mean_14d": 0.05, "error_rate_roll_mean_14d": 25.0
    },

    # --- GROUP 4: CRITICAL FAILURES (Deployment Required) ---
    {
        "desc": "10. Critical: Cooling Fan Death",
        "ambient_temp": 20.0, "avg_peak_temp": 98.0, # <--- Hot despite cool weather
        "voltage_instability": 0.05, "error_rate": 10.0,
        "sessions_today": 5.0, "avg_peak_temp_roll_mean_14d": 95.0, 
        "voltage_instability_roll_mean_14d": 0.05, "error_rate_roll_mean_14d": 8.0
    },
    {
        "desc": "11. Critical: Power Supply Fried",
        "ambient_temp": 20.0, "avg_peak_temp": 55.0, 
        "voltage_instability": 1.5, # <--- Dangerous Voltage
        "error_rate": 25.0,
        "sessions_today": 2.0, "avg_peak_temp_roll_mean_14d": 50.0, 
        "voltage_instability_roll_mean_14d": 1.4, "error_rate_roll_mean_14d": 20.0
    },
    {
        "desc": "12. Critical: Software Deadlock",
        "ambient_temp": 20.0, "avg_peak_temp": 35.0, "voltage_instability": 0.02, 
        "error_rate": 85.0, # <--- Total Crash
        "sessions_today": 5.0, "avg_peak_temp_roll_mean_14d": 35.0, 
        "voltage_instability_roll_mean_14d": 0.02, "error_rate_roll_mean_14d": 80.0
    },
    {
        "desc": "13. Critical: Silent Controller Death",
        "ambient_temp": 20.0, "avg_peak_temp": 30.0, 
        "voltage_instability": 0.9, # <--- Bad Volt
        "error_rate": 45.0,         # <--- Bad Error
        "sessions_today": 1.0,      # <--- Abandoned
        "avg_peak_temp_roll_mean_14d": 30.0, 
        "voltage_instability_roll_mean_14d": 0.8, "error_rate_roll_mean_14d": 40.0
    },
    {
        "desc": "14. Critical: Total Meltdown",
        "ambient_temp": 25.0, "avg_peak_temp": 105.0, 
        "voltage_instability": 1.2, "error_rate": 60.0,
        "sessions_today": 0.0, "avg_peak_temp_roll_mean_14d": 100.0, 
        "voltage_instability_roll_mean_14d": 1.1, "error_rate_roll_mean_14d": 55.0
    },

    # --- GROUP 5: EDGE CASES (Tricky) ---
    {
        "desc": "15. Edge Case: Recovering",
        "ambient_temp": 20.0, "avg_peak_temp": 35.0, # <--- Good Now
        "voltage_instability": 0.01, "error_rate": 0.0,
        "sessions_today": 2.0,
        "avg_peak_temp_roll_mean_14d": 90.0, # <--- History remembers the fire
        "voltage_instability_roll_mean_14d": 0.9, "error_rate_roll_mean_14d": 40.0
    },
    {
        "desc": "16. Edge Case: Impossible Voltage",
        "ambient_temp": 20.0, "avg_peak_temp": 35.0, 
        "voltage_instability": 3.0, # <--- Physically impossible without fire
        "error_rate": 0.0, "sessions_today": 10.0, "avg_peak_temp_roll_mean_14d": 35.0, 
        "voltage_instability_roll_mean_14d": 3.0, "error_rate_roll_mean_14d": 0.0
    },
    {
        "desc": "17. Edge Case: Zero Usage (Just Sitting)",
        "ambient_temp": 20.0, "avg_peak_temp": 20.0, 
        "voltage_instability": 0.01, "error_rate": 0.0,
        "sessions_today": 0.0, # <--- Not used, but healthy
        "avg_peak_temp_roll_mean_14d": 20.0, 
        "voltage_instability_roll_mean_14d": 0.01, "error_rate_roll_mean_14d": 0.0
    },
    {
        "desc": "18. Edge Case: Voltage Spike + Cold",
        "ambient_temp": -10.0, "avg_peak_temp": 5.0, 
        "voltage_instability": 1.1, # <--- Bad
        "error_rate": 5.0, "sessions_today": 2.0,
        "avg_peak_temp_roll_mean_14d": 5.0, 
        "voltage_instability_roll_mean_14d": 1.0, "error_rate_roll_mean_14d": 4.0
    },
    {
        "desc": "19. Edge Case: Error Spike + Hot",
        "ambient_temp": 35.0, "avg_peak_temp": 90.0, 
        "voltage_instability": 0.05, 
        "error_rate": 50.0, # <--- Bad
        "sessions_today": 5.0,
        "avg_peak_temp_roll_mean_14d": 85.0, 
        "voltage_instability_roll_mean_14d": 0.05, "error_rate_roll_mean_14d": 45.0
    },
    {
        "desc": "20. Edge Case: The 'Lemon' (Brand New but Broken)",
        "ambient_temp": 20.0, "avg_peak_temp": 25.0, 
        "voltage_instability": 1.5, "error_rate": 40.0,
        "sessions_today": 0.0, # <--- Never worked
        "avg_peak_temp_roll_mean_14d": 25.0, 
        "voltage_instability_roll_mean_14d": 1.5, "error_rate_roll_mean_14d": 40.0
    }
]

# --- EXECUTE ---
payload = [enrich_scenario(sc) for sc in raw_scenarios]

print(f"🚀 Sending {len(payload)} scenarios to {url}...\n")

try:
    start = time.time()
    response = requests.post(url, json=payload)
    end = time.time()
    print(f"✅ Response received in {end - start:.2f}s\n")
    
    if response.status_code == 200:
        results = response.json()
        
        # Dynamic Formatting for nice columns
        print(f"{'SCENARIO':<40} | {'STATUS':<10} | {'ROOT CAUSE'}")
        print("-" * 90)
        
        for i, res in enumerate(results):
            desc = raw_scenarios[i]['desc']
            status = res['status']
            
            # Clean up Root Cause text
            cause = res.get('root_cause', '-')
            if cause == "Normal Range": cause = "-"
            if len(cause) > 45: cause = cause[:42] + "..."
            
            print(f"{desc:<40} | {status:<10} | {cause}")
            
    else:
        print(f"❌ API Error: {response.text}")

except Exception as e:
    print(f"❌ Connection Error: {e}")