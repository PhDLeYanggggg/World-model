from src.stage42_paper_package_fxfy_refresh import run_stage42_paper_package_fxfy_refresh


if __name__ == "__main__":
    result = run_stage42_paper_package_fxfy_refresh()
    gate = result["stage42_fz_gate"]
    print(f"Stage42-FZ paper package FX/FY refresh: {gate['verdict']} ({gate['passed']}/{gate['total']})")
