from __future__ import annotations

from src.stage19_pipeline import build_wam_data_registry, write_current_state


if __name__ == "__main__":
    write_current_state()
    build_wam_data_registry()

