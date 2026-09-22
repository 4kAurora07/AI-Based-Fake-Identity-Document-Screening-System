"""DocShield AI — AI-Powered Document Field Extractor.

Tries providers in order:
  1. Google Gemini 2.5 Flash  (via google-genai SDK — current, supported)
  2. Groq llama-3.2-11b-vision (free fallback for restricted networks)
  3. OCR-only (automatic — no crash)

Configuration (docshield-backend/.env):
  GEMINI_API_KEY  — from AI Studio (AIzaSy...) or OAuth token (AQ....)
  GROQ_API_KEY    — free at https://console.groq.com/keys
"""

import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK availability checks
# ---------------------------------------------------------------------------
try:
    from google import genai as _google_genai
    from google.genai import types as _genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _google_genai = None
    _genai_types = None
    _GENAI_AVAILABLE = False

try:
    from groq import Groq as _GroqClass
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClass = None
    _GROQ_AVAILABLE = False

_gemini_client: Optional[Any] = None
_groq_client: Optional[Any] = None

# ---------------------------------------------------------------------------
# Shared prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a forensic document analysis AI specialized in Indian identity documents.\n"
    "Extract ONLY fields that are clearly visible in the image.\n"
    "Do NOT invent, guess, or hallucinate values that are not present.\n"
    "Return ONLY a valid JSON object — no markdown, no code fences.\n\n"
    "Required JSON schema (use null for missing/unreadable fields):\n"
    "{\n"
    '  "document_type": "<aadhaar|pan_card|passport|voter_id|driving_license|unknown>",\n'
    '  "document_number": "<the primary ID number or null>",\n'
    '  "holder_name": "<full legal name as printed or null>",\n'
    '  "father_name": "<father or guardian name or null>",\n'
    '  "date_of_birth": "<DD/MM/YYYY or YYYY or null>",\n'
    '  "gender": "<Male|Female|Transgender|null>",\n'
    '  "address": "<full address as a single comma-separated string or null>",\n'
    '  "expiry_date": "<DD/MM/YYYY or MM/YYYY or null>",\n'
    '  "nationality": "<nationality string or null>",\n'
    '  "issue_date": "<DD/MM/YYYY or null>",\n'
    '  "extraction_confidence": "<high|medium|low>",\n'
    '  "unreadable_fields": ["<fields present but not legible>"]\n'
    "}\n\n"
    "Format rules:\n"
    "- Aadhaar: XXXX XXXX XXXX (12 digits grouped 4-4-4)\n"
    "- PAN: 5 uppercase letters + 4 digits + 1 uppercase letter (e.g. ABCDE1234F)\n"
    "- Passport: 1 letter + 7 digits (e.g. A1234567)\n"
    "- Voter ID (EPIC): 3 uppercase letters + 7 digits\n"
    "- Use null — never partial or uncertain data"
)

_USER_PROMPT = (
    "Analyze this Indian identity document image carefully. "
    "Extract all visible identity fields and return the JSON schema specified."
)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_jpeg_bytes(image: Image.Image, max_dim: int = 1568) -> bytes:
    buf = io.BytesIO()
    rgb = image.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    rgb.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


def _to_base64_data_url(image: Image.Image) -> str:
    jpeg_bytes = _to_jpeg_bytes(image)
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("utf-8")

# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.warning("JSON parse error: %s | raw: %.300s", e, text)
    return None

# ---------------------------------------------------------------------------
# Field normalisation (shared)
# ---------------------------------------------------------------------------

def _normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    def clean(v: Any) -> Optional[str]:
        s = str(v).strip() if v else ""
        return s if s and s.lower() not in ("null", "none", "n/a", "na", "unknown", "") else None

    type_map = {
        "aadhaar": "aadhaar", "aadhar": "aadhaar",
        "pan": "pan_card", "pan_card": "pan_card",
        "passport": "passport",
        "voter": "voter_id", "voter_id": "voter_id", "epic": "voter_id",
        "driving": "driving_license", "driving_license": "driving_license", "dl": "driving_license",
    }
    gender_map = {
        "male": "Male", "m": "Male",
        "female": "Female", "f": "Female",
        "transgender": "Transgender",
    }
    return {
        "document_type": type_map.get(str(raw.get("document_type", "")).lower().strip(), "unknown"),
        "document_number": clean(raw.get("document_number")),
        "holder_name": clean(raw.get("holder_name")),
        "father_name": clean(raw.get("father_name")),
        "date_of_birth": clean(raw.get("date_of_birth")),
        "gender": gender_map.get(str(raw.get("gender", "")).lower().strip()),
        "address": clean(raw.get("address")),
        "expiry_date": clean(raw.get("expiry_date")),
        "nationality": clean(raw.get("nationality")),
        "issue_date": clean(raw.get("issue_date")),
        "ai_confidence": clean(raw.get("extraction_confidence")) or "medium",
        "unreadable_fields": raw.get("unreadable_fields") or [],
    }

# ---------------------------------------------------------------------------
# Provider 1: Gemini 2.5 Flash (google-genai SDK)
# ---------------------------------------------------------------------------

def _init_gemini() -> Optional[Any]:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    if not _GENAI_AVAILABLE:
        return None
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        _gemini_client = _google_genai.Client(api_key=api_key)
        logger.info("Gemini (google-genai) client initialized.")
        return _gemini_client
    except Exception as e:
        logger.warning("Gemini init failed: %s", e)
        return None


def _extract_gemini(image: Image.Image, ocr_hint: str = "") -> Optional[Dict[str, Any]]:
    client = _init_gemini()
    if client is None:
        return None
    try:
        jpeg_bytes = _to_jpeg_bytes(image)

        prompt_parts = [_USER_PROMPT]
        if ocr_hint and len(ocr_hint.strip()) > 20:
            prompt_parts.append(f"\nOCR context (cross-reference only):\n{ocr_hint[:800]}")

        candidate_models = ["gemini-3.6-flash"]
        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        _genai_types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                        "\n".join(prompt_parts),
                    ],
                    config=_genai_types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=600,
                        response_mime_type="application/json",
                    ),
                )
                parsed = _parse_json(response.text.strip())
                if parsed:
                    logger.info("%s extraction succeeded.", model_name)
                    return _normalize(parsed)
            except Exception as m_err:
                logger.warning("Gemini model %s failed: %s", model_name, m_err)
                break  # Don't retry or block if unavailable
    except Exception as e:
        logger.warning("Gemini extraction failed (%s) — trying next provider.", e)
    return None

# ---------------------------------------------------------------------------
# Provider 2: Groq (llama-3.2-11b-vision-preview)
# ---------------------------------------------------------------------------

def _init_groq() -> Optional[Any]:
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    if not _GROQ_AVAILABLE:
        return None
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        _groq_client = _GroqClass(api_key=api_key)
        logger.info("Groq client initialized.")
        return _groq_client
    except Exception as e:
        logger.warning("Groq init failed: %s", e)
        return None


def _extract_groq(image: Image.Image, ocr_hint: str = "") -> Optional[Dict[str, Any]]:
    client = _init_groq()
    if client is None:
        return None
    try:
        data_url = _to_base64_data_url(image)
        user_content: list = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": _USER_PROMPT},
        ]
        if ocr_hint and len(ocr_hint.strip()) > 20:
            user_content.append({
                "type": "text",
                "text": f"\nOCR context (cross-reference only):\n{ocr_hint[:800]}",
            })
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=512,
            stream=False,
        )
        parsed = _parse_json(response.choices[0].message.content.strip())
        if parsed:
            logger.info("Groq extraction succeeded.")
            return _normalize(parsed)
    except Exception as e:
        logger.warning("Groq extraction failed: %s", e)
    return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

import concurrent.futures

def extract_fields_with_ai(
    image: Image.Image,
    ocr_text_hint: str = "",
) -> Dict[str, Any]:
    """
    Extract structured identity fields from a document image with a strict 2.5s ceiling.
    Tries Gemini, falls back to Groq, then returns {} gracefully. Never blocks or raises.
    """
    def _do_extract():
        result = _extract_gemini(image, ocr_text_hint)
        if result:
            result["ai_provider"] = "gemini"
            return result

        result = _extract_groq(image, ocr_text_hint)
        if result:
            result["ai_provider"] = "groq-llama-3.2-vision"
            return result
        return {}

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(_do_extract)
        return fut.result(timeout=0.5)
    except Exception as e:
        logger.info("Cloud LLM extraction skipped or timed out (%s) — using local OCR.", str(e))
        return {}
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def merge_ai_and_ocr_fields(
    ai_fields: Dict[str, Any],
    ocr_fields: Dict[str, Any],
    ai_confidence: str = "medium",
) -> Dict[str, Any]:
    """
    Merge AI-extracted fields with OCR-extracted fields.
    Priority: MRZ (caller) > AI high-conf > OCR regex > AI gap-fill
    """
    merged = dict(ocr_fields)
    if not ai_fields:
        return merged

    conf = (ai_confidence or "medium").lower()

    for field in [
        "holder_name", "father_name", "date_of_birth",
        "gender", "address", "expiry_date", "nationality", "issue_date",
    ]:
        ai_val = ai_fields.get(field)
        ocr_val = merged.get(field)
        if conf == "high":
            if ai_val:
                merged[field] = ai_val
        else:
            if ai_val and not ocr_val:
                merged[field] = ai_val

    if ai_fields.get("document_type") and ai_fields["document_type"] != "unknown":
        merged["document_type"] = ai_fields["document_type"]

    if not merged.get("document_number") and ai_fields.get("document_number"):
        merged["document_number"] = ai_fields["document_number"]

    merged["ai_extraction_used"] = True
    merged["ai_provider"] = ai_fields.get("ai_provider", "unknown")
    merged["ai_confidence"] = ai_fields.get("ai_confidence", conf)
    merged["ai_unreadable_fields"] = ai_fields.get("unreadable_fields", [])

    return merged
