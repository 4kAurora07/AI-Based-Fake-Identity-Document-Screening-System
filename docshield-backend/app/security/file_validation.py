"""DocShield AI — Production File Upload Validation & Sanitization Pipeline.

Enforces deep content inspection (magic bytes), decompression bomb prevention,
EXIF stripping, server-side image re-encoding, and secure PDF structural validation.
"""

import io
import os
from typing import Tuple
from PIL import Image, ImageOps, ImageDraw, ImageFont, UnidentifiedImageError

# Configure Pillow decompression bomb protection
# 8000 x 8000 = 64 megapixels maximum allowed image canvas
MAX_PIXELS = 8000 * 8000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class FileValidationError(ValueError):
    """Raised when an uploaded file fails security or format validation."""
    pass


# Magic byte signatures
MAGIC_SIGNATURES = {
    "image/jpeg": [
        b"\xFF\xD8\xFF\xDB",
        b"\xFF\xD8\xFF\xE0",
        b"\xFF\xD8\xFF\xE1",
        b"\xFF\xD8\xFF\xEE",
    ],
    "image/png": [
        b"\x89PNG\r\n\x1a\n",
    ],
    "application/pdf": [
        b"%PDF",
    ],
}

# Known malicious or dangerous headers to explicitly reject immediately
DISALLOWED_SIGNATURES = [
    (b"MZ", "Windows Executable/DLL binary"),
    (b"\x7fELF", "Linux ELF binary"),
    (b"PK\x03\x04", "ZIP / Compressed Archive"),
    (b"<!DOCTYPE", "HTML/XML markup"),
    (b"<svg", "SVG vector graphic containing script vectors"),
    (b"<?xml", "XML document"),
]


def detect_mime_from_magic_bytes(header_bytes: bytes) -> str:
    """Inspects raw initial bytes to determine verified MIME type.

    Does NOT rely on client-supplied Content-Type or file extension.
    """
    if len(header_bytes) < 4:
        raise FileValidationError("Uploaded file is too small or empty.")

    # Check for forbidden binary or script signatures
    for sig, desc in DISALLOWED_SIGNATURES:
        if header_bytes.startswith(sig):
            raise FileValidationError(
                f"Potentially malicious file detected: Content identified as {desc}."
            )

    # Check PDF signature
    if header_bytes.startswith(b"%PDF"):
        return "application/pdf"

    # Check PNG signature
    if header_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    # Check JPEG signatures (SOI marker 0xFF 0xD8)
    if header_bytes.startswith(b"\xFF\xD8"):
        return "image/jpeg"

    raise FileValidationError(
        "Invalid file format: Only genuine JPEG, PNG images, and PDF documents are supported."
    )


def validate_and_reencode_image(
    raw_stream: io.BytesIO,
    max_dimension: int = 8000,
    max_size_bytes: int = 16 * 1024 * 1024,
) -> Tuple[Image.Image, bytes, str]:
    """Validates raw image/PDF data and produces a safe, re-encoded clean image.

    Steps:
    1. Check byte size does not exceed max_size_bytes.
    2. Sniff magic bytes to verify JPEG, PNG, or PDF.
    3. For PDF: validate with pypdf and extract/render document canvas safely.
    4. For Images: inspect dimensions, reject decompression bombs, re-encode to RGB JPEG.

    Returns:
        Tuple of (clean_pil_image, clean_bytes, verified_format)
    """
    raw_stream.seek(0, io.SEEK_END)
    file_size = raw_stream.tell()
    raw_stream.seek(0)

    if file_size <= 0:
        raise FileValidationError("Uploaded file is empty (0 bytes).")

    if file_size > max_size_bytes:
        raise FileValidationError(
            f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds maximum permitted limit ({max_size_bytes / (1024 * 1024):.1f}MB)."
        )

    # Step 1: Magic byte verification
    header = raw_stream.read(32)
    raw_stream.seek(0)
    verified_mime = detect_mime_from_magic_bytes(header)

    # Step 2: Handle PDF Documents
    if verified_mime == "application/pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(raw_stream)
            if len(reader.pages) == 0:
                raise FileValidationError("Uploaded PDF document contains zero pages.")

            # Search first page for embedded document image
            first_page = reader.pages[0]
            clean_image = None

            if len(first_page.images) > 0:
                # Extract primary embedded identity image
                img_data = first_page.images[0]
                with Image.open(io.BytesIO(img_data.data)) as pil_img:
                    clean_image = pil_img.convert("RGB")
            else:
                # Generate document rendering canvas if pure vector/text PDF
                clean_image = Image.new("RGB", (1200, 800), color=(255, 255, 255))
                draw = ImageDraw.Draw(clean_image)
                page_text = first_page.extract_text() or "Document PDF"
                draw.text((40, 40), page_text[:600], fill=(0, 0, 0))

            output_buffer = io.BytesIO()
            clean_image.save(output_buffer, format="JPEG", quality=95, optimize=True)
            return clean_image, output_buffer.getvalue(), "PDF"

        except Exception as e:
            if isinstance(e, FileValidationError):
                raise
            raise FileValidationError(f"PDF validation failed: Malformed or unreadable PDF structure ({str(e)})")

    # Step 3: Handle JPEG and PNG Images
    try:
        with Image.open(raw_stream) as img:
            width, height = img.size

            if width <= 0 or height <= 0:
                raise FileValidationError("Corrupted image: Dimensions are zero or negative.")

            if width > max_dimension or height > max_dimension:
                raise FileValidationError(
                    f"Decompression bomb defense: Image dimensions ({width}x{height}) exceed maximum allowed limit ({max_dimension}x{max_dimension})."
                )

            # Check format reported by Pillow parser
            pillow_format = (img.format or "").upper()
            if pillow_format not in {"JPEG", "JPG", "PNG", "MPO"}:
                raise FileValidationError(
                    f"Disallowed image format '{pillow_format}' detected by parser."
                )

            # Auto-orient based on EXIF before stripping metadata
            oriented_img = ImageOps.exif_transpose(img)
            if oriented_img is None:
                oriented_img = img

            # Convert to standard RGB (drops dangerous color profiles, alpha tricks, paletted code)
            clean_image = oriented_img.convert("RGB")

            # Auto-downscale large images to max 1024px for lightning-fast forensic processing (<1.5s)
            if max(width, height) > 1024:
                scale = 1024.0 / float(max(width, height))
                new_w = max(1, int(width * scale))
                new_h = max(1, int(height * scale))
                clean_image = clean_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    except Image.DecompressionBombError:
        raise FileValidationError(
            "Decompression bomb detected: Image exceeds safe pixel count threshold."
        )
    except UnidentifiedImageError:
        raise FileValidationError("Image decoder failed: Unidentified or corrupted image data.")
    except Exception as e:
        if isinstance(e, FileValidationError):
            raise
        raise FileValidationError("Image processing failed: Malformed image file structure.")

    # Step 4: Re-encode to clean in-memory buffer (completely strips EXIF and embedded steganography/payloads)
    output_buffer = io.BytesIO()
    clean_image.save(output_buffer, format="JPEG", quality=90)
    clean_bytes = output_buffer.getvalue()

    return clean_image, clean_bytes, "JPEG"
