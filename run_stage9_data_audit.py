from __future__ import annotations

from src.evaluation.stage9_data_audit import audit_stage9_data, write_stage9_data_audit


if __name__ == "__main__":
    payload = audit_stage9_data()
    write_stage9_data_audit(payload)
    print(payload)
