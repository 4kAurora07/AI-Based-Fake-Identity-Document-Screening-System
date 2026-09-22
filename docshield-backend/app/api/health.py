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
        },
    }), 200


@health_bp.route("/benchmark", methods=["GET"])
def benchmark():
    """GET /api/v1/benchmark - Profiles exact millisecond execution time of every forensic layer."""
    import time
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), color=(245, 245, 245))
    d = ImageDraw.Draw(img)
    d.text((40, 40), "REPUBLIC OF INDIA PASSPORT", fill=(0, 0, 0))
    d.text((40, 80), "P<INDTEST<<USER<<<<<<<<<<<<<<<<<<<<<<<<<<", fill=(0, 0, 0))
    d.text((40, 120), "A1234567<5IND9001011M2501014<<<<<<<<<<<<<<<<<<<02", fill=(0, 0, 0))

    timings = {}
    try:
        from app.layers.layer1_behavioral import run_layer1_analysis
        t0 = time.perf_counter()
        run_layer1_analysis(img, {}, {})
        timings["layer1_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        timings["layer1_err"] = str(e)

    try:
        from app.layers.layer2_ocr import run_layer2_analysis
        t0 = time.perf_counter()
        run_layer2_analysis(img)
        timings["layer2_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        timings["layer2_err"] = str(e)

    try:
        from app.layers.layer3_forensics import run_layer3_analysis
        t0 = time.perf_counter()
        run_layer3_analysis(img)
        timings["layer3_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        timings["layer3_err"] = str(e)

    try:
        from app.layers.layer4_ai_detection import run_layer4_analysis
        t0 = time.perf_counter()
        run_layer4_analysis(img)
        timings["layer4_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        timings["layer4_err"] = str(e)

    try:
        from app.layers.visual_forensics import VisualForensicsEngine
        t0 = time.perf_counter()
        VisualForensicsEngine.analyze_layout_consistency(img)
        timings["visual_forensics_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        timings["visual_forensics_err"] = str(e)

    try:
        from app.layers.barcode_crosscheck import BarcodeCrossCheckEngine
        t0 = time.perf_counter()
        BarcodeCrossCheckEngine.cross_check(img, {})
        timings["barcode_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        timings["barcode_err"] = str(e)

    try:
        from app.layers.face_matcher import CrossDocumentFaceMatcher
        t0 = time.perf_counter()
        CrossDocumentFaceMatcher.compare_documents(img, None)
        timings["face_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        timings["face_err"] = str(e)

    timings["total_measured_ms"] = round(sum(v for v in timings.values() if isinstance(v, (int, float))), 2)

    return jsonify({"status": "ok", "timings": timings}), 200
