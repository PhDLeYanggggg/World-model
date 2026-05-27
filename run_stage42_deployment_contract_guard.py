from __future__ import annotations

from src.stage42_deployment_contract_guard import run_stage42_deployment_contract_guard


def main() -> None:
    payload = run_stage42_deployment_contract_guard(refresh_readmes=True)
    gate = payload["stage42_ep_gate"]
    print(f"Stage42-EP deployment contract guard complete: {gate['passed']}/{gate['total']} {gate['verdict']}")


if __name__ == "__main__":
    main()
