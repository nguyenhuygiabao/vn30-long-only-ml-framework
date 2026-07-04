from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports" / "figures"


def darken_static_chart(path: Path) -> str:
    image = Image.open(path).convert("RGBA")
    arr = np.array(image).astype(np.uint8)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    visible = alpha > 0

    # Convert white / light-gray figure and axes backgrounds to black.
    light_background = (luma > 175) & (spread < 85) & visible

    # Convert black / dark-gray labels, ticks, titles, and axes to off-white.
    dark_text = (luma < 95) & (spread < 95) & visible

    # Convert neutral mid-gray grid lines / spines to subtle gray.
    mid_gray = (luma >= 95) & (luma <= 175) & (spread < 55) & visible

    rgb[light_background] = [0, 0, 0]
    rgb[dark_text] = [235, 241, 255]
    rgb[mid_gray] = [120, 132, 155]

    arr[:, :, :3] = rgb

    Image.fromarray(arr, mode="RGBA").save(path)
    return "darkened"


def main() -> None:
    if not FIGURES_DIR.exists():
        raise FileNotFoundError(f"Missing figures directory: {FIGURES_DIR}")

    pngs = sorted(FIGURES_DIR.glob("*.png"))

    print()
    print("Darkening static PNG figures safely:")
    for path in pngs:
        result = darken_static_chart(path)
        print(f"- {path.name}: {result}")


if __name__ == "__main__":
    main()
