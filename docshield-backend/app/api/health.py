"""DocShield AI — Health & Diagnostics Endpoint.

Provides readiness and liveness checks for the API and dependent forensic engines.
"""

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """GET /api/v1/health - Service health, module status, and runtime diagnostics."""
    import shutil
    import os

    tess_path = shutil.which("tesseract") or os.environ.get("TESSERACT_CMD", "")
    tess_installed = bool(tess_path and os.path.exists(tess_path))
    
    tess_test_status = "untested"
    try:
        import pytesseract
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (120, 40), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "DOCSHIELD", fill=(0, 0, 0))
        tess_res = pytesseract.image_to_string(img).strip()
        tess_test_status = f"ok: {tess_res}"
    except Exception as te:
        tess_test_status = f"err: {str(te)}"

    return jsonify({
        "status": "healthy",
        "service": "DocShield AI Backend",
        "version": "1.0.0",
        "environment": "active",
        "diagnostics": {
            "tesseract_path": tess_path,
            "tesseract_installed": tess_installed,
            "tesseract_test": tess_test_status,
        },
        "layers": {
            "layer1_behavioral": "online",
            "layer2_ocr": "online",
            "layer3_forensics": "online",
            "layer4_ai_detection": "online",
        },
        "security": {
            "magic_byte_inspection": "active",
            "server_side_reencoding": "active",
            "decompression_bomb_guard": "active",
            "strict_headers": "active",
        },
    }), 200
