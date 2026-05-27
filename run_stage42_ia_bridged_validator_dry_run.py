from src.stage42_ia_bridged_validator_dry_run import run_stage42_ia_bridged_validator_dry_run


if __name__ == "__main__":
    result = run_stage42_ia_bridged_validator_dry_run()
    gate = result["stage42_ib_gate"]
    print(f"Stage42-IB IA bridged validator dry run: {gate['verdict']} ({gate['passed']}/{gate['total']})")
