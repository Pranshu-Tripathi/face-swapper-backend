"""Thin GCS wrapper used by the /merge endpoint.

The shraddha-backend hands us object paths; we read/write the bytes directly
instead of mounting the bucket via FUSE. One singleton client is cheap and
thread-safe per Google's docs.
"""

from __future__ import annotations

import threading

from google.cloud import storage

_client: storage.Client | None = None
_lock = threading.Lock()


def _get_client() -> storage.Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = storage.Client()
    return _client


def read_bytes(bucket: str, object_name: str) -> bytes:
    return _get_client().bucket(bucket).blob(object_name).download_as_bytes()


def write_bytes(
    bucket: str, object_name: str, data: bytes, content_type: str
) -> None:
    blob = _get_client().bucket(bucket).blob(object_name)
    blob.upload_from_string(data, content_type=content_type)
