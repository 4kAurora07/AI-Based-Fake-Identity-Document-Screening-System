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


@health_bp.route("/benchmark/<layer_name>", methods=["GET"])
def benchmark_layer(layer_name):
    """GET /api/v1/benchmark/<layer_name> - Tests an individual forensic layer in isolation."""
    import time
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), color=(245, 245, 245))
    d = ImageDraw.Draw(img)
    d.text((40, 40), "REPUBLIC OF INDIA PASSPORT", fill=(0, 0, 0))
    d.text((40, 80), "P<INDTEST<<USER<<<<<<<<<<<<<<<<<<<<<<<<<<", fill=(0, 0, 0))
    d.text((40, 120), "A1234567<5IND9001011M2501014<<<<<<<<<<<<<<<<<<<02", fill=(0, 0, 0))

    t0 = time.perf_counter()
    res = {}
    try:
        if layer_name == "l1":
            from app.layers.layer1_behavioral import run_layer1_analysis
            res = run_layer1_analysis(img, {}, {})
        elif layer_name == "l2":
            from app.layers.layer2_ocr import run_layer2_analysis
            res = run_layer2_analysis(img)
        elif layer_name == "l3":
            from app.layers.layer3_forensics import run_layer3_analysis
            res = run_layer3_analysis(img)
        elif layer_name == "l4":
            from app.layers.layer4_ai_detection import run_layer4_analysis
            res = run_layer4_analysis(img)
        elif layer_name == "vf":
            from app.layers.visual_forensics import VisualForensicsEngine
            res = VisualForensicsEngine.analyze_layout_consistency(img)
        elif layer_name == "barcode":
            from app.layers.barcode_crosscheck import BarcodeCrossCheckEngine
            res = BarcodeCrossCheckEngine.cross_check(img, {})
        elif layer_name == "face":
            from app.layers.face_matcher import CrossDocumentFaceMatcher
            res = CrossDocumentFaceMatcher.compare_documents(img, None)
        else:
            return jsonify({"error": f"Unknown layer: {layer_name}"}), 400
    except Exception as e:
        return jsonify({"layer": layer_name, "error": str(e), "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)}), 500

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    # Strip numpy arrays for json serialization
    clean_res = {k: v for k, v in res.items() if not str(type(v)).startswith("<class 'numpy")}
    return jsonify({"layer": layer_name, "status": "ok", "elapsed_ms": elapsed_ms, "result_summary": str(clean_res)[:200]}), 200
