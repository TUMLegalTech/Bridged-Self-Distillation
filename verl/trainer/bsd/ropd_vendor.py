"""Load the vendored ROPD implementation without rewriting upstream imports."""

from __future__ import annotations

import sys
from pathlib import Path


def ropd_vendor_root() -> Path:
    """Return the pinned ROPD source root."""

    return Path(__file__).resolve().parents[3] / "third_party" / "ropd_official"


def activate_ropd_vendor() -> Path:
    """Make the pinned upstream ``algo`` package importable."""

    vendor_root = ropd_vendor_root()
    if not (vendor_root / "algo" / "ropd" / "reward_manager.py").is_file():
        raise RuntimeError(f"Missing vendored ROPD source at {vendor_root}")
    vendor_path = str(vendor_root)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    return vendor_root
