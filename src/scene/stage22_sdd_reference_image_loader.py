from __future__ import annotations

from PIL import Image


def load_reference_image(path: str):
    return Image.open(path)


__all__ = ["load_reference_image"]

