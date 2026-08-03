"""
app.services.image_service
=============================

The platform's Image Engine: download -> optimize -> resize -> convert
to WebP -> generate thumbnail -> deduplicate -> store in CDN-backed
object storage.

This module deliberately separates the pure, CPU-bound image
processing (:func:`process_image_bytes`, testable with nothing more
than a byte string in, a dataclass out) from the I/O-bound download
and upload steps (:class:`ImageStorageService`). That separation is
what makes the processing logic verifiable without a network
connection or a real S3 bucket — a real production concern, since
image processing bugs (wrong aspect ratio, corrupted output, wrong
color mode) are exactly the kind of thing you want caught by a fast,
deterministic test rather than discovered after deploying against
real merchant images.

Design notes
------------
* Both Pillow (image processing) and boto3 (S3 upload) are
  synchronous, CPU/IO-bound libraries with no native async API. Each
  is run via ``asyncio.to_thread`` so a single image job doesn't block
  the event loop from servicing other concurrent work — consistent
  with how the feed connector handles its own CPU-bound XML parsing.
* Perceptual hashing uses a dependency-free average-hash (aHash)
  implementation (shrink to 8x8 grayscale, threshold against the mean)
  rather than pulling in an extra ``imagehash`` dependency — aHash is
  sufficient for the platform's stated purpose (catching duplicate
  images from the same or near-identical source), and keeping the
  dependency footprint small matters at this project's scale.
* Images are always converted to RGB before processing (stripping
  alpha/palette modes) and re-encoded as WebP — this normalizes wildly
  inconsistent merchant-supplied formats (some feeds serve GIF, some
  serve CMYK JPEGs, some serve paletted PNGs) into one consistent,
  compact format for storage and display.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import httpx
from PIL import Image, ImageOps

from app.core.config import ObjectStorageSettings

logger = logging.getLogger(__name__)

_DEFAULT_MAX_DIMENSION = 1600
_DEFAULT_THUMBNAIL_DIMENSION = 400
_DEFAULT_WEBP_QUALITY = 85
_AHASH_SIZE = 8  # 8x8 -> a 64-bit perceptual hash


class ImageDownloadError(RuntimeError):
    """Raised when a source image URL cannot be downloaded."""


class ImageProcessingError(RuntimeError):
    """Raised when downloaded bytes cannot be decoded/processed as an
    image (corrupted file, unsupported format, zero-byte response)."""


class ImageStorageError(RuntimeError):
    """Raised when uploading processed image bytes to object storage
    fails."""


@dataclass(frozen=True)
class ProcessedImage:
    """Result of processing one source image: full-size and thumbnail
    WebP bytes ready to upload, plus metadata used for the database
    row and duplicate detection."""

    main_bytes: bytes
    main_width: int
    main_height: int
    thumbnail_bytes: bytes
    thumbnail_width: int
    thumbnail_height: int
    perceptual_hash: str


@dataclass(frozen=True)
class StoredImage:
    """Result of a full download -> process -> upload cycle, ready to
    persist onto a ``ProductImage`` row."""

    cdn_url: str
    thumbnail_cdn_url: str
    width: int
    height: int
    perceptual_hash: str
    size_bytes: int


def compute_average_hash(image: Image.Image) -> str:
    """Compute a 64-bit average hash (aHash) of an image, returned as
    a 16-character hex string.

    Two images with a small Hamming distance between their aHash
    values are very likely near-duplicates (the same product photo at
    different resolutions/compressions, or on a plain background) —
    used by the import pipeline to avoid storing redundant copies of
    what is visually the same image.

    Known limitation (inherent to average-hash, not fixable without a
    different algorithm): a perfectly flat, single-color image always
    hashes to the same value regardless of which color it is, since
    every pixel equals the image's own mean by definition. This is
    irrelevant for real product photography but worth knowing if this
    function is ever reused for synthetic/placeholder images.
    """
    grayscale = image.convert("L").resize(
        (_AHASH_SIZE, _AHASH_SIZE), Image.Resampling.LANCZOS
    )
    pixels = list(grayscale.getdata())
    average = sum(pixels) / len(pixels)

    bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
    # Pack the 64-bit string into a 16-character hex string.
    return f"{int(bits, 2):016x}"


def process_image_bytes(
    raw_bytes: bytes,
    *,
    max_dimension: int = _DEFAULT_MAX_DIMENSION,
    thumbnail_dimension: int = _DEFAULT_THUMBNAIL_DIMENSION,
    webp_quality: int = _DEFAULT_WEBP_QUALITY,
) -> ProcessedImage:
    """Decode, normalize, resize, and re-encode one image as WebP,
    plus a thumbnail and a perceptual hash.

    Pure and synchronous by design (no network, no filesystem, no
    async) — this is what makes it directly unit-testable with a
    handful of in-memory bytes. Callers running this from async code
    should do so via ``asyncio.to_thread`` (see
    :meth:`ImageStorageService.process_and_store`).

    Raises
    ------
    ImageProcessingError
        If ``raw_bytes`` cannot be decoded as an image.
    """
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()  # force full decode now, so corrupt data fails here, not later
    except Exception as exc:
        raise ImageProcessingError(f"Could not decode image data: {exc}") from exc

    # Apply EXIF orientation (common in phone-camera photos merchants
    # upload directly) before stripping metadata, so the image isn't
    # accidentally stored sideways.
    image = ImageOps.exif_transpose(image) or image

    # Normalize to RGB: source images arrive in every mode imaginable
    # (CMYK JPEGs, paletted PNGs, RGBA PNGs with transparency). Flatten
    # transparency onto a white background rather than dropping the
    # alpha channel silently, which would otherwise leave black where
    # transparency used to be for some source formats.
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    perceptual_hash = compute_average_hash(image)

    main_image = ImageOps.contain(image, (max_dimension, max_dimension), Image.Resampling.LANCZOS)
    thumbnail_image = ImageOps.contain(
        image, (thumbnail_dimension, thumbnail_dimension), Image.Resampling.LANCZOS
    )

    main_buffer = io.BytesIO()
    main_image.save(main_buffer, format="WEBP", quality=webp_quality, method=6)

    thumbnail_buffer = io.BytesIO()
    thumbnail_image.save(thumbnail_buffer, format="WEBP", quality=webp_quality, method=6)

    return ProcessedImage(
        main_bytes=main_buffer.getvalue(),
        main_width=main_image.width,
        main_height=main_image.height,
        thumbnail_bytes=thumbnail_buffer.getvalue(),
        thumbnail_width=thumbnail_image.width,
        thumbnail_height=thumbnail_image.height,
        perceptual_hash=perceptual_hash,
    )


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bit-level Hamming distance between two aHash hex strings — the
    standard way to compare two perceptual hashes. A distance of 0
    means identical (after downsampling); the import pipeline treats
    anything below a small threshold (commonly 5) as a likely
    duplicate."""
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


class ImageStorageService:
    """Downloads, processes, and uploads product images to
    CDN-backed object storage.

    Instances are cheap but hold an HTTP client and (lazily) an S3
    client, so — like the database session manager and connector
    registry — application code should obtain the process-wide
    instance via :func:`get_image_storage_service` rather than
    constructing this repeatedly.
    """

    def __init__(self, settings: ObjectStorageSettings) -> None:
        self._settings = settings
        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._s3_client = None  # created lazily; see _get_s3_client

    def _get_s3_client(self):
        """Lazily construct the boto3 S3 client.

        Deferred rather than built in ``__init__`` so importing this
        module (and constructing this class in tests that only
        exercise the pure processing functions) never requires boto3
        to actually reach a real endpoint or have valid-looking
        credentials configured.
        """
        if self._s3_client is None:
            import boto3  # imported here, not at module level, to keep

            # process_image_bytes()/compute_average_hash() usable in
            # environments/tests that don't have boto3 installed at all.
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=str(self._settings.endpoint_url),
                aws_access_key_id=self._settings.access_key_id.get_secret_value(),
                aws_secret_access_key=self._settings.secret_access_key.get_secret_value(),
                region_name=self._settings.region,
            )
        return self._s3_client

    async def _download(self, source_url: str) -> bytes:
        try:
            response = await self._http_client.get(source_url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            raise ImageDownloadError(f"Failed to download image from {source_url}: {exc}") from exc

    async def _upload(self, key: str, data: bytes, content_type: str = "image/webp") -> None:
        client = self._get_s3_client()

        def _put_object() -> None:
            client.put_object(
                Bucket=self._settings.images_bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                CacheControl="public, max-age=31536000, immutable",
            )

        try:
            await asyncio.to_thread(_put_object)
        except Exception as exc:  # boto3 raises various botocore exceptions
            raise ImageStorageError(f"Failed to upload {key} to object storage: {exc}") from exc

    def _build_cdn_url(self, key: str) -> str:
        if self._settings.cdn_base_url:
            base = str(self._settings.cdn_base_url).rstrip("/")
            return f"{base}/{key}"
        # Fall back to the raw endpoint if no CDN is configured yet —
        # documented in ObjectStorageSettings as discouraged for
        # production, but keeps this method usable in early
        # development before a CDN is set up.
        endpoint = str(self._settings.endpoint_url).rstrip("/")
        return f"{endpoint}/{self._settings.images_bucket}/{key}"

    async def process_and_store(self, source_url: str, variant_id: uuid.UUID) -> StoredImage:
        """Full pipeline for one image: download, process (in a worker
        thread), upload both sizes, and return CDN URLs + metadata
        ready to persist as a ``ProductImage`` row.
        """
        raw_bytes = await self._download(source_url)

        processed = await asyncio.to_thread(process_image_bytes, raw_bytes)

        image_id = uuid.uuid4()
        main_key = f"products/{variant_id}/{image_id}.webp"
        thumbnail_key = f"products/{variant_id}/{image_id}_thumb.webp"

        await self._upload(main_key, processed.main_bytes)
        await self._upload(thumbnail_key, processed.thumbnail_bytes)

        return StoredImage(
            cdn_url=self._build_cdn_url(main_key),
            thumbnail_cdn_url=self._build_cdn_url(thumbnail_key),
            width=processed.main_width,
            height=processed.main_height,
            perceptual_hash=processed.perceptual_hash,
            size_bytes=len(processed.main_bytes),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Should be called during
        application shutdown alongside the database session manager's
        own ``dispose()``."""
        await self._http_client.aclose()


_service_instance: Optional[ImageStorageService] = None


def get_image_storage_service() -> ImageStorageService:
    """Return the process-wide :class:`ImageStorageService` singleton,
    constructed from application settings on first access.

    A plain module-level cache (rather than ``lru_cache``) is used
    here deliberately: this service holds an open ``httpx.AsyncClient``
    that must be explicitly closed via :meth:`ImageStorageService.aclose`
    on shutdown, and ``lru_cache`` provides no hook for that — an
    explicit global makes the lifecycle (create once, dispose once)
    visible rather than implicit.
    """
    global _service_instance
    if _service_instance is None:
        from app.core.config import get_settings

        _service_instance = ImageStorageService(get_settings().storage)
    return _service_instance

