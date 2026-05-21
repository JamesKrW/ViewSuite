# multipart.py
"""
Unified multipart helpers for the unified render service.

This module provides both:
- Server-side encoding: build a multipart/mixed response body (JSON meta + N images)
- Client-side decoding: parse a multipart/mixed response body (JSON meta + N images)
- Server-side decoding (optional): parse meta form field and uploaded images

Naming:
- encode_multipart(...)  -> builds multipart/mixed bytes + boundary
- decode_multipart(...)  -> parses multipart/mixed bytes into (meta, images)

Notes:
- The multipart/mixed format here is minimal and tailored to this service.
- Images are handled via Pillow (PIL).
"""

from __future__ import annotations

import io
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    # Optional import: only required on the server side.
    from fastapi import UploadFile
except Exception:  # pragma: no cover
    UploadFile = Any  # type: ignore[misc,assignment]


def numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------
# Client-side / shared: decode multipart/mixed
# ---------------------------------------------------------------------
def _extract_boundary(content_type: str) -> str:
    """
    Extract boundary string from a multipart Content-Type header.

    Args:
        content_type:
            Raw Content-Type header value, e.g.
            'multipart/mixed; boundary="abc123"'.

    Returns:
        Boundary string (without leading '--').

    Raises:
        ValueError:
            If the boundary parameter is missing or empty.
    """
    ct = content_type or ""
    parts = [p.strip() for p in ct.split(";")]
    for p in parts:
        if p.lower().startswith("boundary="):
            b = p.split("=", 1)[1].strip().strip('"')
            if b:
                return b
    raise ValueError(f"Missing boundary in Content-Type: {content_type}")


def decode_multipart(content_type: str, body: bytes) -> Tuple[Dict[str, Any], List[Image.Image]]:
    """
    Decode a multipart/mixed HTTP body into (meta, images).

    Supported parts:
        - application/json  -> metadata dictionary
        - image/*           -> decoded PIL.Image objects

    Args:
        content_type:
            Content-Type header value containing a boundary.
            Must be multipart/* with a valid boundary parameter.
        body:
            Raw HTTP response body bytes.

    Returns:
        Tuple of:
            - meta dict (empty if not present)
            - list of decoded images (may be empty)

    Notes:
        - This parser is intentionally lightweight and does not implement
          all MIME edge cases. It matches the encoder below.
        - Each part payload may include trailing CRLF; the parser tolerates it.
    """
    boundary = _extract_boundary(content_type)
    marker = ("--" + boundary).encode("utf-8")

    meta: Dict[str, Any] = {}
    images: List[Image.Image] = []

    # Split by boundary marker
    chunks = body.split(marker)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk == b"--":
            continue

        # Strip final boundary terminator if present
        if chunk.endswith(b"--"):
            chunk = chunk[:-2].strip()

        header_blob, _, payload = chunk.partition(b"\r\n\r\n")
        if not payload:
            continue

        # Some encoders include trailing CRLF; safe to trim for robustness.
        payload = payload.rstrip(b"\r\n")

        # Parse headers (lightweight)
        headers = header_blob.decode("utf-8", errors="ignore").split("\r\n")
        part_type = ""
        for line in headers:
            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip().lower() == "content-type":
                    part_type = v.strip().lower()

        # JSON metadata part
        if "application/json" in part_type:
            try:
                obj = json.loads(payload.decode("utf-8"))
                meta = obj if isinstance(obj, dict) else {"_meta": obj}
            except Exception:
                meta = {"_meta_raw": payload.decode("utf-8", errors="ignore")}

        # Image part(s)
        elif part_type.startswith("image/"):
            img = Image.open(io.BytesIO(payload)).convert("RGBA")
            images.append(img)

    return meta, images


# ---------------------------------------------------------------------
# Server-side: encode multipart/mixed
# ---------------------------------------------------------------------
def encode_multipart(
    meta: Dict[str, Any],
    images: List[Image.Image],
    *,
    encoded_images: Optional[List[bytes]] = None,
    image_format: str = "PNG",
    image_mime: str = "image/png",
    boundary_prefix: str = "unified_",
) -> Tuple[str, bytes]:
    """
    Encode (meta, images) into a multipart/mixed body.

    Layout:
      - Part name="meta"   Content-Type: application/json; charset=utf-8
      - Part name="images" Content-Type: <image_mime> (repeated)

    Args:
        meta:
            Metadata dictionary to serialize as JSON.
            If meta is empty or None-like, an empty JSON object is serialized.
        images:
            List of PIL Images to encode. May be empty if encoded_images is provided.
        encoded_images:
            Optional list of pre-encoded image bytes. If provided and non-empty,
            this is used directly and PIL encoding is skipped.
        image_format:
            Pillow save() format string, e.g. 'PNG', 'JPEG'.
            Use a lossless format (PNG) if you want exact pixels.
        image_mime:
            MIME type for the image parts, e.g. 'image/png'.
            Must match image_format to avoid confusing clients.
        boundary_prefix:
            Prefix used when generating a unique MIME boundary.

    Returns:
        (boundary, body_bytes)

    Notes:
        - The meta part is always included (even if empty) so clients can
          reliably expect a JSON part.
        - This encoder is designed to be compatible with decode_multipart().
    """
    boundary = f"{boundary_prefix}{uuid.uuid4().hex}"
    crlf = b"\r\n"
    bnd = boundary.encode("utf-8")
    body = bytearray()

    # Meta part
    meta_bytes = json.dumps(meta or {}, ensure_ascii=False).encode("utf-8")
    body += b"--" + bnd + crlf
    body += b'Content-Disposition: form-data; name="meta"; filename="meta.json"' + crlf
    body += b"Content-Type: application/json; charset=utf-8" + crlf + crlf
    body += meta_bytes + crlf

    # Image parts
    if encoded_images:
        image_bytes_list = encoded_images
    else:
        image_bytes_list = []
        for img in images or []:
            buf = io.BytesIO()
            img.save(buf, format=image_format)
            image_bytes_list.append(buf.getvalue())

    for i, img_bytes in enumerate(image_bytes_list):
        body += b"--" + bnd + crlf
        body += f'Content-Disposition: form-data; name="images"; filename="{i}.png"'.encode("utf-8") + crlf
        body += f"Content-Type: {image_mime}".encode("utf-8") + crlf + crlf
        body += img_bytes + crlf

    # End boundary
    body += b"--" + bnd + b"--" + crlf
    return boundary, bytes(body)


# ---------------------------------------------------------------------
# Optional server-side helpers: decode form fields/uploads
# ---------------------------------------------------------------------
def parse_meta_field(meta_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse the optional 'meta' form field used in multipart/form-data requests.

    Expected:
        meta_str is a JSON string.

    Behavior:
        - If meta_str is None/empty: return {}
        - If valid JSON dict: return that dict
        - If valid JSON but not a dict: wrap as {"_meta": <value>}
        - If invalid JSON: wrap as {"_meta_raw": <original string>}

    Args:
        meta_str: Raw form field string for 'meta'.

    Returns:
        Parsed metadata dictionary.
    """
    if not meta_str:
        return {}
    try:
        obj = json.loads(meta_str)
        if isinstance(obj, dict):
            return obj
        return {"_meta": obj}
    except Exception:
        return {"_meta_raw": meta_str}


async def read_images(files: Optional[List[UploadFile]]) -> List[Image.Image]:
    """
    Decode uploaded image files (FastAPI UploadFile) into PIL Images.

    Args:
        files:
            Optional list of uploaded files from a repeated form field
            (commonly named 'images').

    Returns:
        List of decoded RGBA PIL Images. If files is None/empty, returns [].

    Notes:
        - This function is server-only (requires FastAPI UploadFile at runtime).
        - Images are converted to RGBA for consistency across pipelines.
    """
    if not files:
        return []
    imgs: List[Image.Image] = []
    for f in files:
        raw = await f.read()
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        imgs.append(img)
    return imgs
