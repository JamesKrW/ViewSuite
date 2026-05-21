from pathlib import Path
from typing import Optional
from PIL import Image
def safe_open_rgb(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        print(f"Error opening image {path}")
        return None