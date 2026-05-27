from src.stage42_h100_blocker_closure_decision import run_stage42_h100_blocker_closure_decision


if __name__ == "__main__":
    result = run_stage42_h100_blocker_closure_decision()
    gate = result["stage42_gw_gate"]
    print(f"Stage42-GW gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
