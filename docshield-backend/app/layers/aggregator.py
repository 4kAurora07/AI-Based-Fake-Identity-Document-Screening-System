"""DocShield AI — Result Aggregator & Multi-Layer Forensic Fusion Engine.

Runs detection layers concurrently using ThreadPoolExecutor, enforces deterministic
forensic veto rules, compiles evidence-based reason tags, and renders explainable heatmap overlays.
Integrates:
- Layer 1: Behavioral and Device Signals
- Layer 2: OCR & Structural Validation + Barcode/QR Cross-Check
- Layer 3: Image Forensics (ELA, Copy-Move, FFT) + Visual Forensics & Layout Consistency
- Layer 4: Deep Learning Vision Model (EfficientNet-B0)
- Document Source Verification (PDF AcroForms, signatures, EXIF, screenshot indicators)
- Optional Cross-Document Biometric Face Verification
"""

import base64
import concurrent.futures
import io
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from app.layers.layer1_behavioral import run_layer1_analysis
from app.layers.layer2_ocr import run_layer2_analysis
from app.layers.layer3_forensics import run_layer3_analysis
from app.layers.layer4_ai_detection import run_layer4_analysis
from app.layers.document_source import DocumentSourceVerifier
from app.layers.visual_forensics import VisualForensicsEngine
from app.layers.barcode_crosscheck import BarcodeCrossCheckEngine
from app.layers.face_matcher import CrossDocumentFaceMatcher
from app.schemas.analyze_schema import (
    AnalyzeResponse,
    LayerResultsBundle,
    Layer1BehavioralResult,
    Layer2OCRResult,
    Layer3ForensicsResult,
    Layer4AIResult,
    ExtractedDocumentFields,
    BehavioralSignalDetails,
)

logger = logging.getLogger(__name__)


def generate_heatmap_overlay(
    original_img: Image.Image,
    cam_mask: Optional[np.ndarray] = None,
    ela_mask: Optional[np.ndarray] = None,
) -> Optional[str]:
    """Combines CAM and ELA anomaly masks into an overlaid, explainable base64 PNG heatmap.

    Returns:
        Base64-encoded PNG data URL string ('data:image/png;base64,...') or None.
    """
    orig_np = np.array(original_img.convert("RGB"))
    h, w, _ = orig_np.shape

    # Composite anomaly map
    anomaly_map = np.zeros((h, w), dtype=np.float32)

    if cam_mask is not None:
        if cam_mask.shape != (h, w):
            cam_mask = cv2.resize(cam_mask, (w, h))
        anomaly_map += cam_mask.astype(np.float32) * 0.6

    if ela_mask is not None:
        if ela_mask.shape != (h, w):
            ela_mask = cv2.resize(ela_mask, (w, h))
        anomaly_map += ela_mask.astype(np.float32) * 0.4

    # Normalize to 0-255 uint8
    max_val = np.max(anomaly_map)
    if max_val > 10.0:
        norm_map = np.clip((anomaly_map / max_val) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm_map = np.zeros((h, w), dtype=np.uint8)

    # Apply JET colormap (blue = low anomaly, red = high anomaly)
    heatmap_colored = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original image with heatmap where anomaly is pronounced
    alpha = 0.45
    mask_weight = (norm_map / 255.0)[:, :, np.newaxis]
    blended = (orig_np * (1.0 - (mask_weight * alpha)) + heatmap_colored * (mask_weight * alpha)).astype(np.uint8)

    # Encode to base64 PNG
    pil_overlay = Image.fromarray(blended)
    buf = io.BytesIO()
    pil_overlay.save(buf, format="PNG", optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def execute_parallel_analysis(
    image: Image.Image,
    headers: Optional[Dict[str, str]] = None,
    form_data: Optional[Dict[str, Any]] = None,
    tesseract_cmd: str = "",
    model_weights_path: str = "",
    timeout_seconds: float = 18.0,
    raw_bytes: Optional[bytes] = None,
    filename: str = "",
    secondary_image: Optional[Image.Image] = None,
    is_pdf: bool = False,
) -> Dict[str, Any]:
    """Executes all core forensic, OCR, and verification layers concurrently."""
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_l1 = executor.submit(run_layer1_analysis, image, headers, form_data)
        future_l2 = executor.submit(run_layer2_analysis, image, tesseract_cmd)
        future_l3 = executor.submit(run_layer3_analysis, image)
        future_l4 = executor.submit(run_layer4_analysis, image, model_weights_path)
        future_face = executor.submit(CrossDocumentFaceMatcher.compare_documents, image, secondary_image)

        # Document Source Verification has no layer dependencies — run it in parallel too
        def _run_source_verification():
            try:
                if is_pdf and raw_bytes:
                    return DocumentSourceVerifier.inspect_pdf_source(raw_bytes, filename=filename)
                elif raw_bytes:
                    return DocumentSourceVerifier.inspect_image_source(
                        raw_bytes, filename=filename, width=image.width, height=image.height
                    )
                else:
                    return {
                        "status": "UNABLE TO DETERMINE",
                        "confidence": 50.0,
                        "description": "Raw container stream not provided for metadata analysis",
                        "evidence_signals": [],
                        "limitations": "Requires un-sanitized byte stream to inspect EXIF headers.",
                    }
            except Exception as e:
                logger.error("Document Source verification error: %s", str(e))
                return {
                    "status": "UNABLE TO DETERMINE",
                    "confidence": 50.0,
                    "description": f"Source inspection error: {str(e)}",
                    "evidence_signals": [],
                    "limitations": "Error analyzing document source.",
                }

        future_source = executor.submit(_run_source_verification)

        deadline = start_time + timeout_seconds

        def _get_result(fut, default_val, task_name="Task"):
            try:
                remaining = max(0.1, deadline - time.perf_counter())
                return fut.result(timeout=remaining)
            except Exception as e:
                logger.error("%s execution failed or reached deadline: %s", task_name, str(e))
                return default_val

        # Collect results with strict shared deadline
        res_l1 = _get_result(
            future_l1,
            {
                "status": "inconclusive",
                "confidence": 50.0,
                "details": {"is_emulator": False, "is_virtual_camera": False, "is_injection_attack": False, "timestamp_skew_seconds": 0.0, "client_entropy_score": 50.0, "flags": ["Layer 1 timeout/error"]},
            },
            "Layer 1",
        )

        res_l2 = _get_result(
            future_l2,
            {
                "status": "inconclusive",
                "confidence": 50.0,
                "document_type": "unknown",
                "fields": {"document_type": "unknown"},
                "mrz_detected": False,
                "mrz_checksum_valid": None,
                "mrz_format": None,
                "barcode_detected": False,
                "cross_check_matches": True,
                "anomalies": ["Layer 2 processing deadline reached"],
            },
            "Layer 2",
        )

        # Submit dependent tasks as soon as Layer 2 results are available
        def _run_visual_forensics():
            try:
                ocr_lines = res_l2.get("fields", {}).get("raw_lines", [])
                return VisualForensicsEngine.analyze_layout_consistency(image, ocr_lines=ocr_lines)
            except Exception as e:
                logger.error("Visual forensics error: %s", str(e))
                return {
                    "status": "PASS",
                    "confidence": 60.0,
                    "evidence": [],
                    "limitations": "Visual forensics inspection completed with fallback defaults.",
                }

        def _run_barcode_crosscheck():
            try:
                ocr_fields = res_l2.get("fields", {})
                doc_type = res_l2.get("document_type", "unknown")
                return BarcodeCrossCheckEngine.cross_check(image, ocr_fields, doc_type=doc_type)
            except Exception as e:
                logger.error("Barcode cross-check error: %s", str(e))
                return {
                    "status": "NOT DETECTED",
                    "confidence": 50.0,
                    "barcode_detected": False,
                    "details": f"Barcode cross-check error: {str(e)}",
                    "matched_fields": [],
                    "mismatched_fields": [],
                    "limitations": "Barcode decoder encountered an error.",
                }

        future_visual = executor.submit(_run_visual_forensics)
        future_barcode = executor.submit(_run_barcode_crosscheck)

        res_l3 = _get_result(
            future_l3,
            {
                "status": "inconclusive",
                "confidence": 50.0,
                "ela_anomaly_score": 0.0,
                "copy_move_detected": False,
                "copy_move_matches_count": 0,
                "frequency_anomaly_score": 0.0,
                "photo_splicing_detected": False,
                "anomalies": ["Layer 3 processing error or deadline reached"],
                "ela_mask": None,
            },
            "Layer 3",
        )

        res_l4 = _get_result(
            future_l4,
            {
                "status": "inconclusive",
                "confidence": 50.0,
                "model": "EfficientNet-B0",
                "forgery_probability": 50.0,
                "genuine_probability": 50.0,
                "heatmap_generated": False,
                "heatmap_mask": None,
            },
            "Layer 4",
        )

        res_face = _get_result(
            future_face,
            {
                "performed": False,
                "status": "UNABLE TO DETERMINE",
                "similarity_score": 0.0,
                "distance": None,
                "details": "Face matching deadline reached or unperformed.",
                "limitations": "Face matching timed out or encountered an error.",
            },
            "Face Matcher",
        )

        res_source = _get_result(
            future_source,
            {
                "status": "UNABLE TO DETERMINE",
                "confidence": 50.0,
                "description": "Source inspection deadline reached.",
                "evidence_signals": [],
                "limitations": "Error or deadline analyzing document source.",
            },
            "Source Verifier",
        )

        res_visual = _get_result(
            future_visual,
            {
                "status": "PASS",
                "confidence": 60.0,
                "evidence": [],
                "limitations": "Visual forensics inspection completed with fallback defaults.",
            },
            "Visual Forensics",
        )

        res_barcode = _get_result(
            future_barcode,
            {
                "status": "NOT DETECTED",
                "confidence": 50.0,
                "barcode_detected": False,
                "details": "Barcode cross-check completed with fallback defaults.",
                "matched_fields": [],
                "mismatched_fields": [],
                "limitations": "Barcode decoder encountered an error or deadline.",
            },
            "Barcode Cross-Check",
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # -------------------------------------------------------------------------
    # AGGREGATION & DETERMINISTIC VETO ENGINE
    # -------------------------------------------------------------------------
    reason_tags: List[str] = []
    deterministic_fake = False

    # 1. Deterministic Veto Check: Layer 2 MRZ Checksum or Cross-Field Failure
    if res_l2.get("mrz_detected") and res_l2.get("mrz_checksum_valid") is False:
        deterministic_fake = True
        reason_tags.append("Deterministic Failure: ICAO MRZ checksum mismatch (tampered identity fields)")

    if res_l2.get("cross_check_matches") is False:
        deterministic_fake = True
        reason_tags.append("Deterministic Failure: Visual document details do not match MRZ data")

    # 2. Deterministic Veto Check: Barcode / QR Identity Mismatch
    if res_barcode.get("status") == "MISMATCH":
        deterministic_fake = True
        for ev in res_barcode.get("comparison_evidence", []):
            if "MISMATCH" in ev:
                reason_tags.append(f"Barcode Cross-Check: {ev}")
        if not any("Barcode Cross-Check" in t for t in reason_tags):
            reason_tags.append("Deterministic Failure: Printed document ID or Name conflicts with encoded QR payload")

    # 3. Add Layer-specific reason tags
    for anomaly in res_l2.get("anomalies", []):
        if anomaly not in reason_tags:
            reason_tags.append(f"Structural: {anomaly}")

    for anomaly in res_l3.get("anomalies", []):
        reason_tags.append(f"Forensic: {anomaly}")

    for flag in res_l1.get("details", {}).get("flags", []):
        reason_tags.append(f"Behavioral: {flag}")

    raw_ai_class = res_l4.get("raw_class")
    if raw_ai_class == "ai_generated":
        reason_tags.append(f"AI Detection: Vision model classified document as AI-Generated synthetic counterfeit ({res_l4.get('confidence', 0)}% confidence)")
    elif raw_ai_class == "tampered":
        reason_tags.append(f"AI Detection: Vision model detected localized digital tampering ({res_l4.get('confidence', 0)}% confidence)")
    elif res_l4.get("forgery_probability", 0.0) >= 65.0:
        reason_tags.append(f"AI Detection: Neural network flagged forgery ({res_l4['forgery_probability']}%)")

    # 4. Visual Forensics signals
    for ev in res_visual.get("evidence", []):
        if "internally consistent" not in ev and ev not in reason_tags:
            reason_tags.append(f"Visual Forensics: {ev}")

    # 5. Document Source signals
    if res_source.get("status") in ("STRUCTURAL ANOMALY", "POSSIBLE SCAN/SCREENSHOT"):
        reason_tags.append(f"Document Source: {res_source.get('description')}")

    # 6. Biometric Cross-Document Face Match signals
    if res_face.get("performed"):
        if res_face.get("status") == "DIFFERENT":
            reason_tags.append(f"Identity Verification: Facial discrepancy detected between uploaded documents ({res_face.get('similarity_percentage')}%)")
        elif res_face.get("status") == "SAME":
            reason_tags.append(f"Identity Verification: Face match confirmed across documents ({res_face.get('similarity_percentage')}%)")

    # 7. Weighted Scoring Calculation
    # Weights: Layer 2: 30%, Layer 3: 25%, Layer 4: 20%, Visual Forensics: 15%, Layer 1: 10%
    l2_forgery_risk = 100.0 if res_l2.get("status") == "flagged" else (0.0 if res_l2.get("status") == "passed" else 35.0)
    l3_forgery_risk = 100.0 if res_l3.get("status") == "flagged" else (0.0 if res_l3.get("status") == "passed" else 35.0)
    l4_forgery_risk = float(res_l4.get("forgery_probability", 50.0))
    l1_forgery_risk = 100.0 if res_l1.get("status") == "flagged" else (0.0 if res_l1.get("status") == "passed" else 25.0)
    vis_forgery_risk = 70.0 if res_visual.get("status") == "SUSPICIOUS" else 15.0

    # Barcode mismatch adds hard penalty
    if res_barcode.get("status") == "MISMATCH":
        l2_forgery_risk = 100.0

    composite_forgery_score = (
        (l2_forgery_risk * 0.30)
        + (l3_forgery_risk * 0.25)
        + (l4_forgery_risk * 0.20)
        + (vis_forgery_risk * 0.15)
        + (l1_forgery_risk * 0.10)
    )

    # 8. Final Verdict Determination
    if deterministic_fake or composite_forgery_score >= 58.0:
        verdict = "Fake"
        overall_confidence = max(88.0, composite_forgery_score)
    elif composite_forgery_score >= 32.0 or len(reason_tags) > 0:
        verdict = "Suspicious"
        overall_confidence = round(max(composite_forgery_score, 100.0 - composite_forgery_score), 1)
    else:
        verdict = "Genuine"
        overall_confidence = round(100.0 - composite_forgery_score, 1)
        if not reason_tags:
            reason_tags.append("All structural, forensic, typography, and biometric security checks passed")

    # 9. Generate Heatmap Overlay (Prioritize real Grad-CAM from Layer 4)
    heatmap_data_url = res_l4.get("heatmap_base64")
    if not heatmap_data_url:
        cam_mask = res_l4.get("heatmap_mask")
        ela_mask = res_l3.get("ela_mask")
        heatmap_data_url = generate_heatmap_overlay(image, cam_mask=cam_mask, ela_mask=ela_mask)

    # 10. Bundle results into standard schema + raw dicts
    bundle = LayerResultsBundle(
        layer1_behavioral=Layer1BehavioralResult(
            status=res_l1["status"],
            confidence=res_l1["confidence"],
            details=BehavioralSignalDetails(**res_l1.get("details", {})),
        ),
        layer2_ocr=Layer2OCRResult(
            status=res_l2["status"],
            confidence=res_l2["confidence"],
            document_type=res_l2["document_type"],
            fields=ExtractedDocumentFields(**res_l2.get("fields", {})),
            mrz_detected=res_l2["mrz_detected"],
            mrz_checksum_valid=res_l2["mrz_checksum_valid"],
            mrz_format=res_l2["mrz_format"],
            barcode_detected=res_barcode.get("barcode_detected", res_l2["barcode_detected"]),
            cross_check_matches=res_l2["cross_check_matches"] and (res_barcode.get("status") != "MISMATCH"),
            anomalies=res_l2["anomalies"],
        ),
        layer3_forensics=Layer3ForensicsResult(
            status=res_l3["status"],
            confidence=res_l3["confidence"],
            ela_anomaly_score=res_l3["ela_anomaly_score"],
            copy_move_detected=res_l3["copy_move_detected"],
            copy_move_matches_count=res_l3["copy_move_matches_count"],
            frequency_anomaly_score=res_l3["frequency_anomaly_score"],
            photo_splicing_detected=res_l3["photo_splicing_detected"],
            anomalies=res_l3["anomalies"],
        ),
        layer4_ai_detection=Layer4AIResult(
            status=res_l4["status"],
            confidence=res_l4["confidence"],
            model=res_l4["model"],
            forgery_probability=res_l4["forgery_probability"],
            genuine_probability=res_l4["genuine_probability"],
            heatmap_generated=res_l4["heatmap_generated"],
        ),
    )

    clean_layer_results = {
        "layer1": {
            "name": "Behavioral & Device Signals",
            "status": res_l1["status"],
            "confidence": res_l1["confidence"],
            "details": res_l1.get("details", {}),
        },
        "layer2": {
            "name": "OCR & Structural Validation",
            "status": res_l2["status"],
            "confidence": res_l2["confidence"],
            "document_type": res_l2["document_type"],
            "fields": res_l2.get("fields", {}),
            "mrz_detected": res_l2["mrz_detected"],
            "mrz_checksum_valid": res_l2["mrz_checksum_valid"],
            "barcode_detected": res_barcode.get("barcode_detected", res_l2["barcode_detected"]),
            "cross_check_matches": res_l2["cross_check_matches"] and (res_barcode.get("status") != "MISMATCH"),
            "anomalies": res_l2["anomalies"],
        },
        "layer3": {
            "name": "Image Forensics",
            "status": res_l3["status"],
            "confidence": res_l3["confidence"],
            "ela_anomaly_score": res_l3["ela_anomaly_score"],
            "copy_move_detected": res_l3["copy_move_detected"],
            "copy_move_matches_count": res_l3["copy_move_matches_count"],
            "frequency_anomaly_score": res_l3["frequency_anomaly_score"],
            "photo_splicing_detected": res_l3["photo_splicing_detected"],
            "anomalies": res_l3["anomalies"],
        },
        "layer4": {
            "name": "AI / Deep Learning Detection",
            "status": res_l4["status"],
            "confidence": res_l4["confidence"],
            "model": res_l4["model"],
            "forgery_probability": res_l4["forgery_probability"],
            "genuine_probability": res_l4["genuine_probability"],
            "verdict": res_l4.get("verdict", "fake" if res_l4["forgery_probability"] >= 60 else "genuine"),
            "predicted_class": res_l4.get("raw_class") or res_l4.get("details", {}).get("predicted_class"),
            "probabilities": res_l4.get("probabilities") or res_l4.get("details", {}).get("class_probabilities"),
            "details": res_l4.get("details", {}),
        },
        # Advanced verification layers
        "document_source": res_source,
        "visual_forensics": res_visual,
        "barcode_crosscheck": res_barcode,
        "face_match": res_face,
        # Backward compatibility aliases
        "layer1_behavioral": bundle.layer1_behavioral.model_dump(),
        "layer2_ocr": bundle.layer2_ocr.model_dump(),
        "layer3_forensics": bundle.layer3_forensics.model_dump(),
        "layer4_ai_detection": bundle.layer4_ai_detection.model_dump(),
    }

    elapsed = round(elapsed_ms, 2)

    return {
        "verdict": verdict,
        "confidence": round(overall_confidence, 1),
        "reason_tags": reason_tags,
        "heatmap": heatmap_data_url,
        "heatmap_base64": heatmap_data_url,
        "layer_results": clean_layer_results,
        "document_source": res_source,
        "visual_forensics": res_visual,
        "barcode_crosscheck": res_barcode,
        "face_match": res_face,
        "analysis_time_ms": elapsed,
        "processing_time_ms": elapsed,
    }
