from __future__ import annotations

from pathlib import Path

from src.data.stage3_dataset_catalog import write_stage3_data_reports
from src.utils.config import load_config


def main() -> None:
    cfg = load_config("configs/stage3_data.yaml")
    reports = cfg["reports"]
    payload = write_stage3_data_reports(
        reports["data_sources"],
        reports["data_sources_json"],
        reports["variable_schema"],
    )
    print(f"stage3 data catalog written: {Path(reports['data_sources'])}")
    print(f"recommended_next={payload['recommended_next']}")


if __name__ == "__main__":
    main()
