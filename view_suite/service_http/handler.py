# handler.py
"""
Base handler interface for unified HTTP service.
Provides HandlerResult dataclass and BaseHandler abstract base class.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from PIL import Image

@dataclass
class HandlerResult:
    """
    Standard result format for all handlers.

    Attributes:
        meta: Metadata dictionary (can be empty)
        images: List of PIL Images (optional; service will encode)
        encoded_images: List of pre-encoded image bytes (optional; service will passthrough)
        image_format: Image format for encoding when using PIL images (e.g., "PNG")
        image_mime: MIME type for encoded images (e.g., "image/png")
    """
    meta: Dict[str, Any]
    images: Optional[List[Image.Image]] = None
    encoded_images: Optional[List[bytes]] = None
    image_format: Optional[str] = None
    image_mime: Optional[str] = None

class BaseHandler:
    """
    Abstract base class for HTTP service handlers.
    Subclass this to implement custom request handling logic.

    Concurrency control is handled at the service level (not here).
    """

    async def handle(self, meta: Dict[str, Any], images: List[Image.Image]) -> HandlerResult:
        """
        Process a request and return results.

        Args:
            meta: Request metadata dictionary (may be empty)
            images: List of input PIL Images (may be empty)

        Returns:
            HandlerResult with response metadata and images or pre-encoded bytes

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """
        Optional cleanup method called on service shutdown.
        Override this to release resources (connections, file handles, etc.)
        """
        return

class MyHandler(BaseHandler):
    """
    Demo handler implementation.
    ✅ Only modify this class for custom logic.

    Input:
      - meta: dict (may be {})
      - images: List[Image.Image] (may be [])
    Output:
      - same structure, can also be empty
    """
    async def handle(self, meta: Dict[str, Any], images: List[Image.Image]) -> HandlerResult:
        """
        Demo logic: echo input with additional metadata.

        Args:
            meta: Input metadata
            images: Input images

        Returns:
            HandlerResult with augmented metadata and passthrough images
        """
        # Demo logic: echo + add some fields
        out_meta = dict(meta or {})
        out_meta["handled"] = True
        out_meta["num_images_in"] = len(images)

        # Demo: passthrough images
        out_images = images

        return HandlerResult(meta=out_meta, images=out_images)
