"""DocShield AI — Layer 4: AI / Deep Learning Detection Engine.

# MODEL FILE GOES HERE: backend/models/docshield_best_model.pth
# Also checks: docshield-backend/app/ml/weights/docshield_best_model.pth

Wired to the real trained EfficientNet-B0 model checkpoint (3 classes: ai_generated, genuine, tampered)
with Grad-CAM explainable visual heatmaps.
"""

import os
import io
import base64
import logging
from typing import Dict, Any, Union, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.ml.model_loader import get_ai_detector, DEVICE, CLASSES

logger = logging.getLogger(__name__)

# Standard ImageNet normalization for EfficientNet-B0
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==============================================================================
# VERDICT STATE MAPPING
# Map the model's 3 classes onto our UI's verdict states:
#   genuine       -> Genuine
#   ai_generated  -> Fake
#   tampered      -> Suspicious
# ==============================================================================
VERDICT_MAPPING = {
    "genuine": "Genuine",
    "ai_generated": "Fake",
    "tampered": "Suspicious",
}


def map_model_verdict(raw_class: str) -> str:
    """Centralized mapper converting raw PyTorch class names to DocShield UI verdict states."""
    return VERDICT_MAPPING.get(raw_class.lower(), "Suspicious")


def predict(image_path: Union[str, Image.Image]) -> Dict[str, Any]:
    """Runs forward inference on the document image and returns the raw model prediction."""
    model = get_ai_detector()

    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

    class_name = CLASSES[pred_class]
    probs_dict = {CLASSES[i]: round(float(probs[i].item()) * 100.0, 2) for i in range(len(CLASSES))}

    return {
        "verdict": class_name,
        "confidence": round(confidence * 100, 2),
        "predicted_class_index": pred_class,
        "probabilities": probs_dict,
        "ai_generated_prob": probs_dict.get("ai_generated", 0.0),
        "genuine_prob": probs_dict.get("genuine", 0.0),
        "tampered_prob": probs_dict.get("tampered", 0.0),
    }


def generate_heatmap_base64(image_path: Union[str, Image.Image]) -> str:
    """Exact Grad-CAM base64 generator matching prompt specification."""
    heatmap_str, _ = generate_heatmap_with_mask(image_path)
    return heatmap_str


def generate_heatmap_with_mask(
    image_path: Union[str, Image.Image],
    target_class: Optional[int] = None
) -> Tuple[str, Optional[np.ndarray]]:
    """Generates a Grad-CAM heatmap visualization and returns both base64 string and numpy mask."""
    model = get_ai_detector()

    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    orig_w, orig_h = img.size
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized) / 255.0

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = target_class if target_class is not None else torch.argmax(probs).item()

    try:
        targets = [ClassifierOutputTarget(pred_class)]
        with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

        _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        # Resize grayscale cam to original dimensions for mask checks
        resized_mask = cv2.resize((grayscale_cam * 255.0).astype(np.uint8), (orig_w, orig_h))
        return data_url, resized_mask
    except Exception as e:
        logger.warning("Grad-CAM generation error: %s", str(e))
        return "", np.zeros((orig_h, orig_w), dtype=np.uint8)


def run_layer4(image_input: Union[Image.Image, str], weights_path: str = "") -> Dict[str, Any]:
    """Layer 4 public entrypoint: single-pass model inference + Grad-CAM heatmap.

    Previously called predict() then generate_heatmap_with_mask() separately,
    resulting in two redundant forward passes through EfficientNet-B0.
    This merged version reuses the same input tensor for both tasks.
    """
    model = get_ai_detector()

    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    orig_w, orig_h = img.size
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # --- Single forward pass for classification ---
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

    raw_class = CLASSES[pred_class]
    mapped_verdict = map_model_verdict(raw_class)
    probs_dict = {CLASSES[i]: round(float(probs[i].item()) * 100.0, 2) for i in range(len(CLASSES))}
    genuine_prob = probs_dict.get("genuine", 0.0)
    forgery_prob = round(100.0 - genuine_prob, 2)
    is_flagged = raw_class != "genuine"

    # --- Grad-CAM heatmap (reuses same input_tensor, no second forward pass) ---
    heatmap_b64 = ""
    heatmap_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    try:
        img_resized = img.resize((224, 224))
        img_np = np.array(img_resized) / 255.0

        targets = [ClassifierOutputTarget(pred_class)]
        with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
        _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buffer).decode("utf-8")
        heatmap_b64 = f"data:image/png;base64,{b64}"

        # Resize grayscale cam to original dimensions for aggregator mask compositing
        heatmap_mask = cv2.resize((grayscale_cam * 255.0).astype(np.uint8), (orig_w, orig_h))
    except Exception as e:
        logger.warning("Grad-CAM generation error: %s", str(e))

    has_heatmap = bool(heatmap_b64)

    return {
        "status": "flagged" if is_flagged else "passed",
        "verdict": mapped_verdict,
        "raw_class": raw_class,
        "confidence": round(confidence * 100, 2),
        "forgery_probability": forgery_prob,
        "genuine_probability": genuine_prob,
        "heatmap_base64": heatmap_b64,
        "heatmap_mask": heatmap_mask,
        "heatmap_generated": has_heatmap,
        "model": "EfficientNet-B0 (3-Class Trained)",
        "probabilities": probs_dict,
        "details": {
            "predicted_class": raw_class,
            "mapped_verdict": mapped_verdict,
            "class_probabilities": probs_dict,
            "heatmap_available": has_heatmap,
        }
    }


# Alias for compatibility with aggregator imports
run_layer4_analysis = run_layer4


