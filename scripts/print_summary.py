import json
import csv

print("=== 1. WHAT-IF VALIDATION RESULTS ===")
with open("reports/whatif_validation.json", "r", encoding="utf-8") as f:
    print(json.dumps(json.load(f), indent=2))

print("\n=== 2. APPROXIMATE FOOD ITEMS ===")
with open("data/external/indian_meal_templates.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["is_approximate"].strip().lower() == "true":
            print(f"- {row['food']}: {row['notes']}")

print("\n=== 3. ALERT USABLE STATUS FOR ALL EVENTS ACROSS ALL HORIZONS ===")
with open("reports/operating_thresholds.json", "r", encoding="utf-8") as f:
    events = json.load(f)["events"]
    for k, v in events.items():
        print(
            f"{k:28s} | alert_usable={str(v.get('alert_usable')):5s} | recall={v.get('recall_mean', 0.0):.2f} | prec={v.get('precision_mean', 0.0):.2f} | reliable={str(v.get('is_reliable')):5s}"
        )
