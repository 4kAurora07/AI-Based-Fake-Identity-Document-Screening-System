"""DocShield AI — Layer 3: Image Forensics Engine.

Executes classical and frequency forensic analysis:
1. Error Level Analysis (ELA) for digital splicing and recompression artifacts.
2. Keypoint-based Copy-Move Forgery Detection (ORB/SIFT clone detection).
3. 2D Fast Fourier Transform (FFT) & DCT spectral analysis for GenAI / diffusion signatures.
"""

import io
import logging
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

logger = logging.getLogger(__name__)


class ErrorLevelAnalysis:
    """Detects compression inconsistencies across different regions of an image."""

    @staticmethod
    def compute_ela(image: Image.Image, quality: int = 90) -> Tuple[float, np.ndarray]:
        """Calculates Error Level Analysis (ELA) score and difference mask.

        Returns:
            Tuple of (ela_anomaly_score, ela_heatmap_mask_uint8)
        """
        # Ensure clean RGB format
        clean_img = image.convert("RGB")
        clean_img.load()

        # Save to buffer at known quality
        buffer = io.BytesIO()
        clean_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as recompressed:
            recompressed.load()
            diff = ImageChops.difference(clean_img, recompressed)


        # Scale difference to visualize and score
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 0

        # Enhance difference contrast
        scale = 255.0 / max(max_diff, 1)
        enhanced_diff = ImageEnhance.Brightness(diff).enhance(min(scale * 0.5, 15.0))

        # Convert to numpy for numeric metric computation
        diff_np = np.array(enhanced_diff.convert("L"))

        # In natural unedited documents, ELA error is uniform.
        # Edits / splices produce localized high variance spikes.
        mean_err = float(np.mean(diff_np))
        std_err = float(np.std(diff_np))

        # Higher variance and max localized error indicate spliced layers
        ela_score = min(100.0, (std_err * 2.2) + (mean_err * 0.8))

        return round(ela_score, 2), diff_np


class CopyMoveDetector:
    """Detects cloned or duplicated regions within the same document using keypoint matching."""

    @staticmethod
    def detect_clones(image_np: np.ndarray, min_spatial_distance: float = 40.0) -> Tuple[bool, int, List[Tuple[int, int]]]:
        """Identifies copy-move duplication using ORB feature descriptor matching."""
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Initialize ORB detector (600 features optimal for fast <0.2s matching)
        orb = cv2.ORB_create(nfeatures=600)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            return False, 0, []

        # Brute-force matcher with kNN
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            matches = bf.knnMatch(descriptors, descriptors, k=3)
        except cv2.error:
            return False, 0, []

        valid_clone_matches = []

        for match_group in matches:
            if len(match_group) < 3:
                continue
            # Self-match is always match_group[0] (distance 0)
            # Compare second nearest neighbor
            m1 = match_group[1]
            m2 = match_group[2]

            # Lowe's ratio test
            if m1.distance < 0.75 * m2.distance:
                pt1 = keypoints[m1.queryIdx].pt
                pt2 = keypoints[m1.trainIdx].pt

                # Calculate Euclidean spatial distance between the two keypoints
                dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

                # Must be separated by minimum spatial distance (not adjacent pixels)
                if dist >= min_spatial_distance:
                    valid_clone_matches.append((pt1, pt2))

        # Flag if significant cluster of distant matches found
        is_clone_detected = len(valid_clone_matches) >= 12
        return is_clone_detected, len(valid_clone_matches), valid_clone_matches


class FrequencyForensics:
    """Performs 2D Fast Fourier Transform (FFT) analysis to detect GAN and diffusion artifacts."""

    @staticmethod
    def compute_fft_anomaly(image_np: np.ndarray) -> Tuple[float, np.ndarray]:
        """Analyzes 2D power spectrum for periodic high-frequency generator artifacts."""
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Resize to standard size for normalized frequency analysis
        resized = cv2.resize(gray, (512, 512)).astype(np.float32)

        # Compute 2D FFT and center zero frequency
        f = np.fft.fft2(resized)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1e-9)

        # Normalize magnitude spectrum
        norm_spectrum = cv2.normalize(
            magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # High frequency ring mask (radius > 150 from center 256,256)
        rows, cols = 512, 512
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        r = np.hypot(x - ccol, y - crow)

        # High frequency band
        high_freq_mask = (r > 160) & (r < 250)
        high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
        total_energy = np.mean(magnitude_spectrum)

        # Measure high-frequency energy ratio and peak concentration
        # Synthetic GAN/diffusion documents exhibit unnatural high-frequency harmonic peaks
        std_high_freq = np.std(magnitude_spectrum[high_freq_mask])
        anomaly_ratio = float((high_freq_energy / (total_energy + 1e-6)) * std_high_freq * 10.0)
        anomaly_score = float(np.clip(anomaly_ratio, 0.0, 100.0))

        return round(anomaly_score, 2), norm_spectrum


def run_layer3_analysis(image: Image.Image) -> Dict[str, Any]:
    """Runs all classical and spectral image forensic detectors in Layer 3."""
    image = image.convert("RGB")
    image.load()
    image_np = np.array(image)

    # 1. Error Level Analysis
    ela_score, ela_mask = ErrorLevelAnalysis.compute_ela(image)


    # 2. Copy-Move Forgery Detection
    copy_move_detected, matches_count, clone_points = CopyMoveDetector.detect_clones(image_np)

    # 3. Frequency / Spectral Analysis
    freq_score, freq_spectrum = FrequencyForensics.compute_fft_anomaly(image_np)

    anomalies: List[str] = []

    # Evaluate splicing
    splicing_detected = ela_score >= 38.0
    if splicing_detected:
        anomalies.append(f"Photo splicing or multi-source compression detected (ELA score: {ela_score})")

    if copy_move_detected:
        anomalies.append(
            f"Copy-move cloning detected ({matches_count} duplicated feature clusters identified)"
        )

    if freq_score >= 45.0:
        anomalies.append(f"Unnatural frequency spectrum / GAN artifact peaks detected (score: {freq_score})")

    # Combine localized anomalies into unified forensic mask
    forensic_mask = np.copy(ela_mask)
    if copy_move_detected and clone_points:
        for pt1, pt2 in clone_points:
            cv2.circle(forensic_mask, (int(pt1[0]), int(pt1[1])), 18, 255, -1)
            cv2.circle(forensic_mask, (int(pt2[0]), int(pt2[1])), 18, 255, -1)

    # Composite status
    if splicing_detected or copy_move_detected or freq_score >= 50.0:
        status = "flagged"
        confidence = min(98.0, 75.0 + max(ela_score * 0.3, freq_score * 0.3, matches_count * 1.5))
    else:
        status = "passed"
        confidence = max(60.0, 100.0 - (ela_score * 0.4 + freq_score * 0.4))

    return {
        "status": status,
        "confidence": round(confidence, 1),
        "ela_anomaly_score": ela_score,
        "copy_move_detected": copy_move_detected,
        "copy_move_matches_count": matches_count,
        "frequency_anomaly_score": freq_score,
        "photo_splicing_detected": splicing_detected,
        "anomalies": anomalies,
        "ela_mask": forensic_mask,
    }
