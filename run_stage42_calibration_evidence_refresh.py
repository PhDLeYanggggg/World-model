from src.stage42_calibration_evidence_refresh import build_calibration_evidence_refresh


if __name__ == "__main__":
    result = build_calibration_evidence_refresh()
    gate = result["stage42_ad_gate"]
    print(f"Stage42-AD calibration evidence refresh: {gate['passed']} / {gate['total']} {gate['verdict']}")
