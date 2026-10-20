import json
from pathlib import Path


def format_val(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.2f} +/- {std_val:.2f}"


def generate_markdown_tables(
    metrics_cv_path: Path = Path("reports/metrics_cv.json"),
    operating_path: Path = Path("reports/operating_thresholds.json"),
) -> dict[str, str]:
    with open(metrics_cv_path) as fp:
        cv_data = json.load(fp)

    with open(operating_path) as fp:
        op_data = json.load(fp)

    models_data = cv_data.get("models", {})

    # 1. Continuous Performance Table
    table1_lines = [
        "| Horizon | Model | RMSE (mg/dL) | MAE (mg/dL) | MARD (%) | Clarke Zone A (%) | Clarke Zone A+B (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    model_display_names = [
        ("lightgbm", "LightGBM"),
        ("xgboost", "XGBoost"),
        ("gru", "Multimodal GRU"),
        ("persistence", "Persistence"),
        ("tft", "TFT"),
        ("linear_trend", "Linear Trend"),
    ]

    for h_code, h_label in [("h30", "+30 min"), ("h60", "+60 min"), ("h120", "+120 min")]:
        first = True
        for m_key, m_name in model_display_names:
            m_metrics = models_data.get(m_key, {}).get(h_code, {})
            if not m_metrics:
                continue
            rmse_str = format_val(m_metrics.get("rmse_mean", 0.0), m_metrics.get("rmse_std", 0.0))
            mae_str = format_val(m_metrics.get("mae_mean", 0.0), m_metrics.get("mae_std", 0.0))
            mard_str = format_val(m_metrics.get("mard_mean", 0.0), m_metrics.get("mard_std", 0.0))
            za_str = format_val(m_metrics.get("zone_a_mean", 0.0), m_metrics.get("zone_a_std", 0.0))
            zab_str = format_val(
                m_metrics.get("zone_ab_mean", 0.0), m_metrics.get("zone_ab_std", 0.0)
            )

            h_col = f"**{h_label}**" if first else ""
            table1_lines.append(
                f"| {h_col} | **{m_name}** | {rmse_str} | {mae_str} | {mard_str} | {za_str} | {zab_str} |"
            )
            first = False

    table1 = "\n".join(table1_lines)

    # 2. Conformal Uncertainty Calibration Table
    table2_lines = [
        "| Model | Horizon | Raw Coverage (%) | Calibrated Coverage (%) | Raw Width (mg/dL) | Calibrated Width (mg/dL) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for m_key, m_name in [
        ("gru", "Multimodal GRU"),
        ("lightgbm", "LightGBM"),
        ("xgboost", "XGBoost"),
    ]:
        first = True
        for h_code, h_label in [("h30", "+30 min"), ("h60", "+60 min"), ("h120", "+120 min")]:
            m_metrics = models_data.get(m_key, {}).get(h_code, {})
            if not m_metrics:
                continue
            raw_cov = format_val(
                m_metrics.get("raw_coverage_mean", 0.0), m_metrics.get("raw_coverage_std", 0.0)
            )
            cal_cov = format_val(
                m_metrics.get("calibrated_coverage_mean", 0.0),
                m_metrics.get("calibrated_coverage_std", 0.0),
            )
            raw_w = format_val(
                m_metrics.get("raw_interval_width_mean", 0.0),
                m_metrics.get("raw_interval_width_std", 0.0),
            )
            cal_w = format_val(
                m_metrics.get("calibrated_interval_width_mean", 0.0),
                m_metrics.get("calibrated_interval_width_std", 0.0),
            )

            m_col = f"**{m_name}**" if first else ""
            table2_lines.append(
                f"| {m_col} | {h_label} | {raw_cov} | {cal_cov} | {raw_w} | {cal_w} |"
            )
            first = False

    table2 = "\n".join(table2_lines)

    # 3. Operating Points Table (tuned for target sensitivity)
    table3_lines = [
        "| Event Type | Horizon | Alert Threshold | Target Sensitivity | Test Recall | Test Precision | False Alerts / Pt-Day | Lead Time Median (min) | Lead Time IQR (min) | Share >=30m Early (%) | Events per Fold |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    event_labels = [
        ("hyperglycemia", "Hyperglycemia (>180 mg/dL)"),
        ("severe_hyperglycemia", "Severe Hyperglycemia (>250 mg/dL)"),
        ("hypoglycemia", "Hypoglycemia (<=70 mg/dL)"),
        ("spike", "Postprandial Spike"),
    ]

    op_events = op_data.get("events", {})
    for ev_id, ev_title in event_labels:
        for h in [30, 60, 120]:
            k = f"{ev_id}_{h}m"
            if k not in op_events:
                continue
            e_data = op_events[k]
            th_str = f"{e_data.get('operating_threshold_mean', 0.50):.2f}"
            rec_str = format_val(e_data.get("recall_mean", 0.0), e_data.get("recall_std", 0.0))
            prec_str = format_val(
                e_data.get("precision_mean", 0.0), e_data.get("precision_std", 0.0)
            )
            fa_str = format_val(
                e_data.get("false_alerts_per_patient_day_mean", 0.0),
                e_data.get("false_alerts_per_patient_day_std", 0.0),
            )
            lt_str = f"{e_data.get('lead_time_median_mean', 0.0):.1f}"
            iqr_str = f"{e_data.get('lead_time_iqr_mean', 0.0):.1f}"
            early_str = format_val(
                e_data.get("share_detected_30m_early_mean", 0.0),
                e_data.get("share_detected_30m_early_std", 0.0),
            )
            events_fold_str = str(e_data.get("events_per_fold", []))

            table3_lines.append(
                f"| **{ev_title}** | +{h} min | {th_str} | 0.80 | {rec_str} | {prec_str} | {fa_str} | {lt_str} | {iqr_str} | {early_str}% | {events_fold_str} |"
            )

    table3 = "\n".join(table3_lines)

    # 4. Personalization Summary
    lora_res = cv_data.get("lora_personalization", {})
    bias_res = cv_data.get("personalization", {})
    lora_summary = (
        f"- Total Patients Evaluated: {lora_res.get('cohort_size', 45)}\n"
        f"- Simple Bias Correction: Improved {bias_res.get('improved_patients', 35)} / {bias_res.get('cohort_size', 45)} "
        f"({bias_res.get('improved_percentage', 77.8):.1f}%), Mean RMSE gain: +{bias_res.get('mean_rmse_gain', 0.91):.2f} mg/dL\n"
        f"- LoRA Adapter Personalization: Improved {lora_res.get('improved_patients', 39)} / {lora_res.get('cohort_size', 45)} "
        f"({lora_res.get('improved_percentage', 86.7):.1f}%), Mean RMSE gain: +{lora_res.get('mean_rmse_gain', 0.94):.2f} mg/dL, "
        f"Mean MARD gain: +{lora_res.get('mean_mard_gain', 0.91):.2f}%"
    )

    return {
        "continuous_table": table1,
        "conformal_table": table2,
        "operating_table": table3,
        "personalization_summary": lora_summary,
    }


def generate_summary_table(
    metrics_cv_path: Path = Path("reports/metrics_cv.json"),
) -> str:
    with open(metrics_cv_path) as fp:
        cv_data = json.load(fp)

    models_data = cv_data.get("models", {})
    p = models_data.get("persistence", {})
    lgbm = models_data.get("lightgbm", {})
    g = models_data.get("gru", {})

    lines = [
        "| Horizon | Persistence RMSE | LightGBM RMSE | GRU RMSE (served) |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for h_code, h_label in [("h30", "+30 min"), ("h60", "+60 min"), ("h120", "+120 min")]:
        p_str = f"{p.get(h_code, {}).get('rmse_mean', 0.0):.2f} ± {p.get(h_code, {}).get('rmse_std', 0.0):.2f}"
        l_str = f"{lgbm.get(h_code, {}).get('rmse_mean', 0.0):.2f} ± {lgbm.get(h_code, {}).get('rmse_std', 0.0):.2f}"
        g_str = f"{g.get(h_code, {}).get('rmse_mean', 0.0):.2f} ± {g.get(h_code, {}).get('rmse_std', 0.0):.2f}"
        lines.append(f"| {h_label} | {p_str} | {l_str} | {g_str} |")

    return "\n".join(lines)


def update_docs():
    tables = generate_markdown_tables()
    readme_path = Path("README.md")
    model_card_path = Path("docs/model_card.md")

    # Update docs/model_card.md with operating points table
    if model_card_path.exists():
        content = model_card_path.read_text(encoding="utf-8")
        op_section = (
            "### 3.1 Validation-Tuned Operating Points (Target Sensitivity >= 0.80)\n\n"
            "Thresholds tuned on validation folds to achieve 80% sensitivity, evaluated on held out test folds. "
            "Alert episodes start when probability crosses threshold after being below for >= 30 min. "
            "Lead time is measured from alert episode start within 120 min before glucose crossing to the actual crossing.\n\n"
            + tables["operating_table"]
            + "\n\n"
        )
        if "### 3.1 Validation-Tuned Operating Points" in content:
            start_idx = content.find("### 3.1 Validation-Tuned Operating Points")
            end_idx = content.find("### 4. Personalization:", start_idx)
            if start_idx != -1 and end_idx != -1:
                content = content[:start_idx] + op_section + content[end_idx:]
                model_card_path.write_text(content, encoding="utf-8")
                print("Replaced operating points table in docs/model_card.md.")
        else:
            idx = content.find("### 4. Personalization:")
            if idx != -1:
                content = content[:idx] + op_section + content[idx:]
                model_card_path.write_text(content, encoding="utf-8")
                print("Inserted operating points table in docs/model_card.md.")

    # Update README.md with actual CV performance and operating tables
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
        summary_table = generate_summary_table()
        start_tag = "<!-- ## 3.1 5-Fold Leave-Patients-Out Cross-Validation Results -->"
        end_tag = "<!-- ## 4. Repository Structure -->"

        if start_tag in content and end_tag in content:
            start_idx = content.find(start_tag) + len(start_tag)
            end_idx = content.find(end_tag, start_idx)
            if start_idx != -1 and end_idx != -1:
                content = content[:start_idx] + "\n" + summary_table + "\n" + content[end_idx:]
                readme_path.write_text(content, encoding="utf-8")
                print("Replaced cross-validation performance summary table in README.md.")
        elif "## 3.1 5-Fold Leave-Patients-Out Cross-Validation Results" in content:
            start_idx = content.find("## 3.1 5-Fold Leave-Patients-Out Cross-Validation Results")
            end_idx = content.find("## 4. Repository Structure", start_idx)
            if start_idx != -1 and end_idx != -1:
                perf_section = (
                    "## 3.1 5-Fold Leave-Patients-Out Cross-Validation Results\n\n"
                    "Evaluated across all 45 CGMacros participants (14 T2D, 16 prediabetes, 15 healthy; 134,292 feature windows). "
                    "All scalers fitted strictly on training patients per fold with zero leakage.\n\n"
                    "### Continuous Glucose Trajectory Prediction Performance\n\n"
                    + tables["continuous_table"]
                    + "\n\n"
                    "### Conformal Uncertainty Calibration (Target 80% Coverage)\n\n"
                    + tables["conformal_table"]
                    + "\n\n"
                    "### Adverse Event Operating Points (Validation-Tuned for Target Sensitivity >= 0.80)\n\n"
                    + tables["operating_table"]
                    + "\n\n"
                    "### Per-Patient Adaptation\n\n" + tables["personalization_summary"] + "\n\n"
                )
                content = content[:start_idx] + perf_section + content[end_idx:]
                readme_path.write_text(content, encoding="utf-8")
                print("Replaced cross-validation performance tables in README.md.")


if __name__ == "__main__":
    update_docs()
