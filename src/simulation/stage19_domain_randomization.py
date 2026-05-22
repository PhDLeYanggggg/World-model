from __future__ import annotations


def domain_randomization_config(seed: int = 19) -> dict:
    return {"seed": seed, "noise": "bounded", "layout_randomization": True, "sensor_dropout": True}

