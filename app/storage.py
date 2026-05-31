from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+\.(jpg|jpeg|png)$")


def new_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{secrets.token_hex(3)}.jpg"


def safe_resolve(storage_root: Path, subdir: str, file_id: str) -> Path:
    if not _SAFE_ID.match(file_id):
        raise ValueError(f"invalid id: {file_id!r}")
    path = storage_root / subdir / file_id
    if not path.is_file():
        raise FileNotFoundError(file_id)
    return path
