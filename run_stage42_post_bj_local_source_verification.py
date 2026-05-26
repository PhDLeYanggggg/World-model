from src.stage42_post_bj_local_source_verification import run_stage42_post_bj_local_source_verification


if __name__ == "__main__":
    result = run_stage42_post_bj_local_source_verification()
    gate = result["stage42_bk_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
