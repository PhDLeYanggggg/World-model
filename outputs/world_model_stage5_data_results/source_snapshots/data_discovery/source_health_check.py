from __future__ import annotations

from typing import Dict, Iterable, List


def health_check_rows(rows: Iterable[Dict], network: bool = False) -> List[Dict]:
    # Default is offline-safe: validate registry completeness without pinging websites.
    checks = []
    for row in rows:
        missing = [key for key in ["dataset_name", "official_url", "download_status", "license", "loader_status"] if not row.get(key)]
        checks.append(
            {
                "dataset_name": row.get("dataset_name"),
                "official_url": row.get("official_url"),
                "registry_complete": not missing,
                "missing_fields": ",".join(missing),
                "network_checked": network,
                "health_status": "not_checked_network" if not network else "network_check_not_implemented",
            }
        )
    return checks
