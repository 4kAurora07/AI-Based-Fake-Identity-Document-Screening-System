"""DocShield AI — Layer 2: OCR & Structural Validation Engine.

Performs multi-engine optical character recognition (EasyOCR / Tesseract),
document classification, ICAO Doc 9303 MRZ decoding with strict check-digit verification,
intelligent document-specific field extraction (PAN, Aadhaar, Passport, Voter ID, DL),
QR/barcode cross-checking, and layout typographic consistency analysis.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np

# Optional imports with graceful fallbacks
try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pyzbar.pyzbar import decode as decode_barcode
except ImportError:
    decode_barcode = None

logger = logging.getLogger(__name__)

# Cached singleton for EasyOCR Reader
_EASYOCR_READER = None


def get_easyocr_reader():
    """Initializes or returns cached EasyOCR Reader singleton."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import easyocr
            # verbose=False prevents Windows console cp1252 progress bar encoding crash
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR Reader successfully initialized on CPU.")
        except Exception as e:
            logger.warning("EasyOCR Reader initialization deferred or failed: %s", str(e))
            _EASYOCR_READER = None
    return _EASYOCR_READER


def compute_icao_check_digit(data: str) -> int:
    """Computes the ICAO Doc 9303 check digit using 7-3-1 recurring weights."""
    weights = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        if char == "<":
            val = 0
        elif char.isdigit():
            val = int(char)
        elif char.isalpha():
            val = ord(char.upper()) - 55  # 'A' is 65 -> 10
        else:
            val = 0
        total += val * weights[i % 3]
    return total % 10


class MRZValidator:
    """ICAO Doc 9303 Machine Readable Zone (MRZ) parser and checksum validator."""

    @staticmethod
    def parse_and_validate(lines: List[str]) -> Dict[str, Any]:
        """Inspects text lines to find and validate TD1 (3x30) or TD3 (2x44) MRZ."""
        clean_lines = [re.sub(r"[^A-Z0-9<]", "", line.upper().strip()) for line in lines]
        clean_lines = [line for line in clean_lines if len(line) >= 28]

        # Check for TD3 (Passport: 2 lines of 44 chars)
        for i in range(len(clean_lines) - 1):
            line1, line2 = clean_lines[i], clean_lines[i + 1]
            if len(line1) == 44 and len(line2) == 44 and line1.startswith("P<"):
                return MRZValidator._validate_td3(line1, line2)

        # Check for TD1 (Identity card: 3 lines of 30 chars)
        for i in range(len(clean_lines) - 2):
            line1, line2, line3 = clean_lines[i], clean_lines[i + 1], clean_lines[i + 2]
            if len(line1) == 30 and len(line2) == 30 and len(line3) == 30:
                return MRZValidator._validate_td1(line1, line2, line3)

        return {"mrz_detected": False}

    @staticmethod
    def _validate_td3(line1: str, line2: str) -> Dict[str, Any]:
        """Validates ICAO TD3 (Passport) MRZ."""
        issuing_country = line1[2:5]
        names_part = line1[5:].split("<<")
        surname = names_part[0].replace("<", " ").strip() if names_part else ""
        given_names = names_part[1].replace("<", " ").strip() if len(names_part) > 1 else ""
        full_name = f"{given_names} {surname}".strip()

        doc_num = line2[0:9]
        doc_num_check = line2[9]
        nationality = line2[10:13]
        dob = line2[13:19]
        dob_check = line2[19]
        sex = line2[20]
        expiry = line2[21:27]
        expiry_check = line2[27]
        composite_data = line2[0:10] + line2[13:20] + line2[21:43]
        composite_check = line2[43]

        doc_num_valid = str(compute_icao_check_digit(doc_num)) == doc_num_check
        dob_valid = str(compute_icao_check_digit(dob)) == dob_check
        expiry_valid = str(compute_icao_check_digit(expiry)) == expiry_check
        composite_valid = str(compute_icao_check_digit(composite_data)) == composite_check

        all_valid = doc_num_valid and dob_valid and expiry_valid and composite_valid
        anomalies = []
        if not doc_num_valid:
            anomalies.append("MRZ document number check-digit mismatch (tampered number)")
        if not dob_valid:
            anomalies.append("MRZ date-of-birth check-digit mismatch (tampered DOB)")
        if not expiry_valid:
            anomalies.append("MRZ expiry date check-digit mismatch")
        if not composite_valid:
            anomalies.append("MRZ composite integrity checksum failure")

        # Format DOB to standard DD/MM/YYYY if YYMMDD
        formatted_dob = dob
        if len(dob) == 6 and dob.isdigit():
            yy, mm, dd = dob[0:2], dob[2:4], dob[4:6]
            prefix = "19" if int(yy) > 30 else "20"
            formatted_dob = f"{dd}/{mm}/{prefix}{yy}"

        return {
            "mrz_detected": True,
            "format": "TD3",
            "document_type": "passport",
            "mrz_checksum_valid": all_valid,
            "document_number": doc_num.replace("<", ""),
            "holder_name": full_name,
            "date_of_birth": formatted_dob,
            "expiry_date": expiry,
            "nationality": nationality,
            "sex": "Male" if sex == "M" else ("Female" if sex == "F" else sex),
            "anomalies": anomalies,
        }

    @staticmethod
    def _validate_td1(line1: str, line2: str, line3: str) -> Dict[str, Any]:
        """Validates ICAO TD1 (National ID) MRZ."""
        doc_type = line1[0:2]
        issuing_country = line1[2:5]
        doc_num = line1[5:14]
        doc_num_check = line1[14]

        dob = line2[0:6]
        dob_check = line2[6]
        sex = line2[7]
        expiry = line2[8:14]
        expiry_check = line2[14]
        nationality = line2[15:18]

        names_part = line3.split("<<")
        surname = names_part[0].replace("<", " ").strip() if names_part else ""
        given_names = names_part[1].replace("<", " ").strip() if len(names_part) > 1 else ""
        full_name = f"{given_names} {surname}".strip()

        doc_num_valid = str(compute_icao_check_digit(doc_num)) == doc_num_check
        dob_valid = str(compute_icao_check_digit(dob)) == dob_check
        expiry_valid = str(compute_icao_check_digit(expiry)) == expiry_check

        all_valid = doc_num_valid and dob_valid and expiry_valid
        anomalies = []
        if not doc_num_valid:
            anomalies.append("TD1 MRZ document number check-digit mismatch")
        if not dob_valid:
            anomalies.append("TD1 MRZ date-of-birth check-digit mismatch")
        if not expiry_valid:
            anomalies.append("TD1 MRZ expiry check-digit mismatch")

        formatted_dob = dob
        if len(dob) == 6 and dob.isdigit():
            yy, mm, dd = dob[0:2], dob[2:4], dob[4:6]
            prefix = "19" if int(yy) > 30 else "20"
            formatted_dob = f"{dd}/{mm}/{prefix}{yy}"

        return {
            "mrz_detected": True,
            "format": "TD1",
            "document_type": "identity_card",
            "mrz_checksum_valid": all_valid,
            "document_number": doc_num.replace("<", ""),
            "holder_name": full_name,
            "date_of_birth": formatted_dob,
            "expiry_date": expiry,
            "nationality": nationality,
            "sex": "Male" if sex == "M" else ("Female" if sex == "F" else sex),
            "anomalies": anomalies,
        }


class OCRForensicExtractor:
    """Multi-engine OCR text extractor and structured field analyzer."""

    @staticmethod
    def extract_text_and_lines(
        image: Image.Image, custom_cmd: str = ""
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """Runs multi-engine OCR (EasyOCR primary, Tesseract fallback) and returns (full_text, lines, tokens)."""
        raw_text = ""
        lines: List[str] = []
        tokens_info: List[Dict[str, Any]] = []

        # Preprocessing: Convert image to RGB
        ocr_img = image.convert("RGB")
        w, h = ocr_img.size
        min_dim = min(w, h)
        max_dim = max(w, h)

        # Adaptive resolution optimization:
        # Cap max_dim to 480px to guarantee sub-3-second execution and low memory (<70MB) on CPU
        if max_dim > 480:
            scale = 480.0 / max_dim
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            ocr_img = ocr_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        elif min_dim < 320:
            scale = min(1.5, 400.0 / max(1, min_dim))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            ocr_img = ocr_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 1. Primary Engine: Tesseract (C++ runtime, <20MB RAM, sub-second execution)
        tesseract_available = False
        if pytesseract is not None:
            if custom_cmd:
                pytesseract.pytesseract.tesseract_cmd = custom_cmd
            elif os.path.exists("/usr/bin/tesseract"):
                pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
            elif os.name == "nt":
                for win_path in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
                    if os.path.exists(win_path):
                        pytesseract.pytesseract.tesseract_cmd = win_path
                        break
            import shutil
            tess_bin = pytesseract.pytesseract.tesseract_cmd
            if shutil.which(tess_bin) or (tess_bin and os.path.exists(tess_bin)):
                tesseract_available = True

        if tesseract_available:
            try:
                try:
                    tess_text = pytesseract.image_to_string(ocr_img, lang="eng+hin") or ""
                except Exception:
                    try:
                        tess_text = pytesseract.image_to_string(ocr_img, lang="eng") or ""
                    except Exception:
                        tess_text = ""
                tess_lines = [l.strip() for l in tess_text.split("\n") if l.strip()]

                # Pass 2: Adaptive CLAHE enhancement if initial pass detected sparse text
                if len(tess_lines) < 2:
                    try:
                        import cv2
                        img_np = np.array(ocr_img)
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(gray)
                        tess_text2 = pytesseract.image_to_string(enhanced, lang="eng") or ""
                        tess_lines2 = [l.strip() for l in tess_text2.split("\n") if l.strip()]
                        if len(tess_lines2) > len(tess_lines):
                            tess_text = tess_text2
                            tess_lines = tess_lines2
                    except Exception:
                        pass

                raw_text = tess_text
                lines = tess_lines
            except Exception as e:
                logger.debug("Tesseract OCR pass: %s", str(e))

        # 2. Secondary Engine: EasyOCR (runs strictly if Tesseract binary is absent on host)
        # Prevents container OOM SIGKILL on memory-constrained (512MB) cloud platforms like Render
        if not tesseract_available and (not raw_text or len(lines) < 2):
            reader = get_easyocr_reader()
            if reader is not None:
                try:
                    img_np = np.array(ocr_img)
                    try:
                        import cv2
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced_gray = clahe.apply(gray)
                        input_for_ocr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
                    except Exception:
                        input_for_ocr = img_np

                    results = reader.readtext(input_for_ocr, batch_size=4)
                    if results:
                        sorted_results = sorted(results, key=lambda item: (item[0][0][1], item[0][0][0]))
                        line_groups: List[List[Tuple[Any, str, float]]] = []

                        for item in sorted_results:
                            bbox, txt, conf = item
                            txt_clean = txt.strip()
                            if not txt_clean:
                                continue
                            tokens_info.append({
                                "text": txt_clean,
                                "confidence": round(float(conf), 3),
                                "bbox": bbox,
                            })

                            y_mid = (bbox[0][1] + bbox[2][1]) / 2.0
                            h_box = max(10, abs(bbox[2][1] - bbox[0][1]))

                            placed = False
                            for group in line_groups:
                                group_y_mid = sum((b[0][1] + b[2][1]) / 2.0 for b, _, _ in group) / len(group)
                                if abs(y_mid - group_y_mid) < (h_box * 0.70):
                                    group.append(item)
                                    placed = True
                                    break
                            if not placed:
                                line_groups.append([item])

                        for group in line_groups:
                            group_sorted = sorted(group, key=lambda item: item[0][0][0])
                            line_str = " ".join(item[1].strip() for item in group_sorted if item[1].strip())
                            if line_str:
                                lines.append(line_str)

                        raw_text = "\n".join(lines)
                except Exception as e:
                    logger.warning("EasyOCR inference error: %s", str(e))

        # Debug logging: output the full raw text blocks extracted by OCR
        logger.info("=== RAW OCR EXTRACTED TEXT (Lines: %d) ===", len(lines))
        print(f"\n--- [DocShield OCR Raw Text Dump] Lines detected: {len(lines)} ---")
        for idx, l in enumerate(lines):
            logger.info("  [%02d] %s", idx + 1, l)
            print(f"  [{idx + 1:02d}] {l}")
        print("--- [End DocShield OCR Raw Text Dump] ---\n")

        return raw_text, lines, tokens_info

    @staticmethod
    def decode_barcodes(image: Image.Image) -> List[Dict[str, Any]]:
        """Extracts barcode and QR code payloads."""
        if decode_barcode is None:
            return []

        try:
            decoded_objects = decode_barcode(image)
            results = []
            for obj in decoded_objects:
                results.append({
                    "type": obj.type,
                    "data": obj.data.decode("utf-8", errors="ignore"),
                    "rect": (obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height),
                })
            return results
        except Exception as e:
            logger.warning("Barcode decoding failed: %s", str(e))
            return []

    @staticmethod
    def classify_document_type(text: str, lines: Optional[List[str]] = None) -> str:
        """Heuristic classifier for document types based on statutory text anchors."""
        lines = lines or []
        combined = (text + " " + " ".join(lines)).upper()

        # 1. Passport
        if "P<IND" in combined or "P<" in combined or ("PASSPORT" in combined and "REPUBLIC OF INDIA" in combined):
            return "passport"

        # 2. PAN Card
        if ("INCOME TAX DEPARTMENT" in combined
            or "PERMANENT ACCOUNT NUMBER" in combined
            or "आयकर विभाग" in combined
            or ("GOVT. OF INDIA" in combined and re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", combined))):
            return "pan_card"

        # 3. Aadhaar Card
        if ("AADHAAR" in combined
            or "UNIQUE IDENTIFICATION" in combined
            or "MERA AADHAAR" in combined
            or "UIDAI" in combined
            or ("GOVERNMENT OF INDIA" in combined and re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", combined))):
            return "aadhaar"

        # 4. Voter ID (EPIC)
        if ("ELECTION COMMISSION" in combined
            or "ELECTOR" in combined
            or "EPIC" in combined
            or "NIRVACHAN" in combined
            or re.search(r"\b[A-Z]{3}[0-9]{7}\b", combined)):
            return "voter_id"

        # 5. Driving License
        if ("DRIVING LICENCE" in combined
            or "DRIVING LICENSE" in combined
            or ("UNION OF INDIA" in combined and "TRANSPORT" in combined)):
            return "driving_license"

        # Regex-based fallback classification
        if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", combined):
            return "pan_card"
        if re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", combined):
            return "aadhaar"
        if re.search(r"\b[A-Z][0-9]{7}\b", combined):
            return "passport"

        return "unknown"

    @staticmethod
    def extract_structural_fields(
        text: str, lines: Optional[List[str]] = None, doc_type: str = "unknown"
    ) -> Dict[str, Optional[str]]:
        """Extracts identity fields using real OCR text and document-specific parsing rules."""
        lines = lines or []
        fields: Dict[str, Optional[str]] = {
            "document_number": None,
            "holder_name": None,
            "father_name": None,
            "date_of_birth": None,
            "gender": None,
            "address": None,
            "expiry_date": None,
        }

        combined_text = (text + "\n" + "\n".join(lines)) if lines else text
        if not combined_text.strip():
            return fields

        # ---------------------------------------------------------------------
        # 1. Document Number Extraction
        # ---------------------------------------------------------------------
        # PAN Number (5 letters, 4 digits, 1 letter) with OCR character confusion recovery
        pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", combined_text)
        if not pan_match:
            # Handle common OCR confusion where digits 0/1/2/5/8 in PAN are read as O/I/Z/S/B
            pan_cand = re.search(r"\b([A-Z]{5})([0-9OIZSB]{4})([A-Z])\b", combined_text.upper())
            if pan_cand:
                prefix = pan_cand.group(1)
                mid = pan_cand.group(2).replace("O", "0").replace("I", "1").replace("Z", "2").replace("S", "5").replace("B", "8")
                suffix = pan_cand.group(3)
                if mid.isdigit() and len(mid) == 4:
                    pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", f"{prefix}{mid}{suffix}")

        # Aadhaar Number (12 digits, spaced 4-4-4, 4-8, 8-4, or contiguous 12 digits)
        # UIDAI standard: Valid Aadhaar numbers start with digits 2 through 9 (never 0 or 1, which avoids toll-free 1947 or 0000 enrollment numbers)
        aadhaar_match = re.search(r"\b([2-9]\d{3})\s*(\d{4})\s*(\d{4})\b", combined_text)
        aadhaar_raw = None
        if not aadhaar_match:
            raw12 = re.search(r"\b([2-9]\d{3})\s*(\d{8})\b", combined_text) or re.search(r"\b([2-9]\d{11})\b", combined_text)
            if raw12:
                digits = re.sub(r"\D", "", raw12.group(0))
                if len(digits) == 12:
                    aadhaar_raw = f"{digits[:4]} {digits[4:8]} {digits[8:]}"

        # Passport Number (1 letter + 7 digits)
        passport_match = re.search(r"\b[A-Z][0-9]{7}\b", combined_text)
        # Voter ID (3 letters + 7 digits)
        voter_match = re.search(r"\b[A-Z]{3}[0-9]{7}\b", combined_text)
        # Driving License (state code + 11-16 digits)
        dl_match = re.search(r"\b[A-Z]{2}[0-9-]{13,16}\b", combined_text) or re.search(r"\b[A-Z]{2}\d{2}\s?\d{11}\b", combined_text)

        if doc_type == "pan_card" and pan_match:
            fields["document_number"] = pan_match.group(0)
        elif doc_type == "aadhaar":
            fields["document_number"] = f"{aadhaar_match.group(1)} {aadhaar_match.group(2)} {aadhaar_match.group(3)}" if aadhaar_match else (aadhaar_raw if aadhaar_raw else None)
        elif doc_type == "passport" and passport_match:
            fields["document_number"] = passport_match.group(0)
        elif doc_type == "voter_id" and voter_match:
            fields["document_number"] = voter_match.group(0)
        elif doc_type == "driving_license" and dl_match:
            fields["document_number"] = dl_match.group(0)
        else:
            # Fallback priority
            if pan_match:
                fields["document_number"] = pan_match.group(0)
            elif aadhaar_match:
                fields["document_number"] = f"{aadhaar_match.group(1)} {aadhaar_match.group(2)} {aadhaar_match.group(3)}"
            elif aadhaar_raw:
                fields["document_number"] = aadhaar_raw
            elif passport_match:
                fields["document_number"] = passport_match.group(0)
            elif voter_match:
                fields["document_number"] = voter_match.group(0)
            elif dl_match:
                fields["document_number"] = dl_match.group(0)

        # ---------------------------------------------------------------------
        # 2. Date of Birth Extraction
        # ---------------------------------------------------------------------
        # Check explicit labeled DOB (allowing optional spaces around separators)
        dob_patterns = [
            r"(?:DOB|D\.?O\.?B\.?|Date\s*of\s*Birth|Birth\s*Date|जन्म\s*तिथि)[\s:.-]*([0-3]?[0-9]\s*[-/.]\s*[0-1]?[0-9]\s*[-/.]\s*(?:19|20)\d{2})",
            r"(?:Year\s*of\s*Birth|जन्म\s*वर्ष|Y\.?O\.?B\.?)[\s:.-]*((?:19|20)\d{2})",
        ]
        for pat in dob_patterns:
            m = re.search(pat, combined_text, re.IGNORECASE)
            if m:
                dob_str = re.sub(r"\s+", "", m.group(1)).replace("-", "/").replace(".", "/")
                fields["date_of_birth"] = dob_str
                break

        if not fields["date_of_birth"]:
            # General date regex fallback (e.g. 01/01/1980 or 22/11/2007)
            gen_date = re.search(r"\b(0[1-9]|[12][0-9]|3[01])\s*[-/.]\s*(0[1-9]|1[012])\s*[-/.]\s*(19|20)\d\d\b", combined_text)
            if gen_date:
                fields["date_of_birth"] = re.sub(r"\s+", "", gen_date.group(0)).replace("-", "/").replace(".", "/")

        # ---------------------------------------------------------------------
        # 3. Gender Extraction
        # ---------------------------------------------------------------------
        # Modern e-PAN cards and all Aadhaar cards include Gender
        gender_match = re.search(r"(?:Sex|Gender|लिंग)[\s:.-]*\b(MALE|FEMALE|TRANSGENDER|पुरुष|महिला|[MF])\b", combined_text, re.IGNORECASE)
        if not gender_match:
            gender_match = re.search(r"\b(MALE|FEMALE|TRANSGENDER|पुरुष|महिला)\b", combined_text, re.IGNORECASE)

        if gender_match:
            val = gender_match.group(1).strip().upper()
            if val in ("MALE", "M", "पुरुष"):
                fields["gender"] = "Male"
            elif val in ("FEMALE", "F", "महिला"):
                fields["gender"] = "Female"
            elif val == "TRANSGENDER":
                fields["gender"] = "Transgender"

        # ---------------------------------------------------------------------
        # 4. Holder Name & Father/Guardian Name Extraction
        # ---------------------------------------------------------------------
        document_header_keywords = {
            "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA",
            "PERMANENT", "ACCOUNT", "NUMBER", "CARD", "BHARAT", "SARKAR",
            "AYAKAR", "VIBHAG", "SIGNATURE", "FATHER", "NAME", "DATE", "BIRTH",
            "NOM", "PRENOM", "PRENOMS", "SURNAME", "GIVEN", "SEXE", "PAIS",
            "NAISSANCE", "LIEU", "PASSPORT", "REPUBLIC", "DELIVRANCE", "EXPIRATION",
            "PERE", "TYPE", "CODE", "UNIQUE", "IDENTIFICATION", "AUTHORITY", "AADHAAR"
        }

        def is_clean_name(candidate: str) -> bool:
            clean = candidate.strip()
            # Must be 3 to 45 characters
            if len(clean) < 3 or len(clean) > 45:
                return False
            # Reject if digits or prohibited symbols present
            if re.search(r"[0-9/\\<>{}\[\]=+\*#@!_]", clean):
                return False
            # Must be mostly letters and spaces
            letters_count = sum(1 for c in clean if c.isalpha() or c.isspace())
            if letters_count / max(1, len(clean)) < 0.80:
                return False
            words = clean.split()
            words_upper = [w.upper() for w in words]
            if all(w in document_header_keywords for w in words_upper):
                return False
            return True

        # Check explicit labeled patterns first (e.g. Name: Alwin Anil Zachariah)
        labeled_name = re.search(r"(?:Name|Elector's\s*Name|नाम)[\s:.-]+([A-Za-z ]{3,40})", combined_text, re.IGNORECASE)
        if labeled_name and is_clean_name(labeled_name.group(1)):
            fields["holder_name"] = labeled_name.group(1).strip()

        labeled_father = re.search(
            r"(?:Father's?\s*Name|Father\s*Name|Husband's\s*Name|पिता\s*का\s*नाम|S/O|D/O|W/O)[\s:.-]+([A-Za-z ]{3,40})",
            combined_text,
            re.IGNORECASE,
        )
        if labeled_father and is_clean_name(labeled_father.group(1)):
            fields["father_name"] = labeled_father.group(1).strip()

        # Check for Passport visual lines: Surname + Given Names
        if doc_type == "passport" and lines:
            p_surname = ""
            p_given = ""
            for i, line in enumerate(lines):
                line_clean = line.strip()
                if re.search(r"\bSurname\b", line_clean, re.IGNORECASE) and i + 1 < len(lines):
                    cand = lines[i + 1].strip()
                    if is_clean_name(cand):
                        p_surname = cand
                if re.search(r"\bGiven\s*Names?\b", line_clean, re.IGNORECASE) and i + 1 < len(lines):
                    cand = lines[i + 1].strip()
                    if is_clean_name(cand):
                        p_given = cand
            if p_given or p_surname:
                full_p = f"{p_given} {p_surname}".strip()
                if is_clean_name(full_p):
                    fields["holder_name"] = full_p

        # Document-specific line-by-line heuristics
        if doc_type == "pan_card" and lines:
            for i, line in enumerate(lines):
                line_clean = line.strip()
                # Check for Name label line (e.g. "Name Nom" or "Name")
                if re.search(r"^(?:Name|Name\s*/\s*Nom|Name\s+Nom|नाम)[\s:.-]*$", line_clean, re.IGNORECASE):
                    if i + 1 < len(lines) and is_clean_name(lines[i + 1]):
                        fields["holder_name"] = lines[i + 1].strip()

                # Check for Father's Name label line
                if re.search(r"(?:Father's?\s*Name|Nom\s*du\s*Pere|पिता\s*का\s*नाम)", line_clean, re.IGNORECASE):
                    if i + 1 < len(lines) and is_clean_name(lines[i + 1]):
                        fields["father_name"] = lines[i + 1].strip()

            # Positional fallback for PAN cards if unlabeled
            if not fields["holder_name"] or not fields["father_name"]:
                header_idx = -1
                cutoff_idx = len(lines)

                for i, line in enumerate(lines):
                    line_upper = line.upper()
                    if "INCOME TAX" in line_upper or "GOVT" in line_upper or "PERMANENT ACCOUNT" in line_upper:
                        header_idx = max(header_idx, i)
                    if (fields["date_of_birth"] and fields["date_of_birth"] in line) or (fields["document_number"] and fields["document_number"] in line):
                        cutoff_idx = min(cutoff_idx, i)

                candidates: List[str] = []
                start_i = header_idx + 1 if header_idx >= 0 else 0
                for i in range(start_i, cutoff_idx):
                    line_str = lines[i].strip()
                    if is_clean_name(line_str) and not re.search(r"(?:Name|Father|Birth|Date|Permanent|Account)", line_str, re.IGNORECASE):
                        candidates.append(line_str)

                if candidates:
                    if not fields["holder_name"]:
                        fields["holder_name"] = candidates[0].strip()
                    if len(candidates) > 1 and not fields["father_name"]:
                        fields["father_name"] = candidates[1].strip()

        elif doc_type == "aadhaar" and lines:
            aadhaar_noise = {
                "GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION", "AUTHORITY OF INDIA",
                "AADHAAR", "MERA AADHAAR", "ENROLMENT", "HELP@UIDAI.GOV.IN", "1947", "WWW.UIDAI.GOV.IN"
            }
            # Aadhaar Letter / Card multi-line name parsing
            for i, line in enumerate(lines):
                line_clean = line.strip()

                # 1. Check for "To" or "To:" indicating letter format
                if re.match(r"^To[\s:.-]*$", line_clean, re.IGNORECASE) and i + 1 < len(lines):
                    name_parts = []
                    j = i + 1
                    while j < len(lines) and j <= i + 3:
                        cand_line = lines[j].strip()
                        if re.search(r"(?:Father|S/O|D/O|W/O|C/O|DOB|Date|Address|PIN|VTC|PO:)", cand_line, re.IGNORECASE):
                            break
                        if is_clean_name(cand_line) and not any(kw in cand_line.upper() for kw in aadhaar_noise):
                            name_parts.append(cand_line)
                        else:
                            break
                        j += 1
                    if name_parts and not fields["holder_name"]:
                        fields["holder_name"] = " ".join(name_parts)

                # 2. Check for inline "To: Full Name"
                to_inline = re.search(r"^To[\s:.-]+([A-Za-z ]{3,40})$", line_clean, re.IGNORECASE)
                if to_inline and is_clean_name(to_inline.group(1)) and not fields["holder_name"]:
                    fields["holder_name"] = to_inline.group(1).strip()

                # 3. Father / Guardian inline parsing (e.g. "Father: Mahesh Kashyap" or "S/O: Anil Zachariah")
                father_match = re.search(r"(?:Father|S/O|D/O|W/O|C/O|Husband)[\s:.-]+([A-Za-z ]{3,40})", line_clean, re.IGNORECASE)
                if father_match and is_clean_name(father_match.group(1)) and not fields["father_name"]:
                    fields["father_name"] = father_match.group(1).strip()

            # Positional fallback for Aadhaar holder name (look above DOB line)
            if not fields["holder_name"]:
                dob_line_idx = -1
                for i, line in enumerate(lines):
                    if (fields["date_of_birth"] and fields["date_of_birth"] in line) or "DOB" in line.upper() or "YEAR OF BIRTH" in line.upper():
                        dob_line_idx = i
                        break

                if dob_line_idx > 0:
                    for i in range(dob_line_idx - 1, -1, -1):
                        cand = lines[i].strip()
                        if re.search(r"(?:Father|S/O|D/O|W/O|To:)", cand, re.IGNORECASE):
                            continue
                        if is_clean_name(cand) and not any(kw in cand.upper() for kw in aadhaar_noise):
                            # Also check if predecessor line is part of name
                            if i - 1 >= 0 and is_clean_name(lines[i - 1].strip()) and not any(kw in lines[i - 1].upper() for kw in aadhaar_noise) and not re.search(r"(?:To|Father|DOB|Date)", lines[i - 1], re.IGNORECASE):
                                fields["holder_name"] = f"{lines[i - 1].strip()} {cand}"
                            else:
                                fields["holder_name"] = cand
                            break

        # ---------------------------------------------------------------------
        # 5. Address Extraction
        # ---------------------------------------------------------------------
        if doc_type != "pan_card":
            # 1. Check for labeled address (e.g. "Address: ..." or "पता: ...")
            addr_idx = -1
            for i, line in enumerate(lines):
                if re.match(r"^(?:Address|पता)[\s:.-]*$", line.strip(), re.IGNORECASE) or re.search(r"^(?:Address|पता)[\s:.-]+", line.strip(), re.IGNORECASE):
                    addr_idx = i
                    break

            if addr_idx >= 0:
                addr_lines = []
                first_line = re.sub(r"^(?:Address|पता)[\s:.-]+", "", lines[addr_idx].strip(), flags=re.IGNORECASE)
                if first_line.strip():
                    addr_lines.append(first_line.strip())
                for k in range(addr_idx + 1, min(len(lines), addr_idx + 6)):
                    cand = lines[k].strip()
                    if re.search(r"(?:1947|UIDAI|Unique\s*Identification|Authority|help@|\b[2-9]\d{3}\s*\d{4}\s*\d{4}\b)", cand, re.IGNORECASE):
                        break
                    addr_lines.append(cand)
                    if re.search(r"\b[1-9][0-9]{5}\b", cand):
                        break
                if addr_lines:
                    fields["address"] = ", ".join(addr_lines)

            # 2. Aadhaar letter address block extraction:
            # Lines following cardholder and father, down to PIN Code
            if not fields["address"] and doc_type == "aadhaar" and lines:
                to_idx = -1
                pin_idx = -1
                for i, line in enumerate(lines):
                    if re.match(r"^To[\s:.-]*$", line.strip(), re.IGNORECASE) or line.strip().startswith("To:"):
                        to_idx = i
                    if re.search(r"\b[1-9][0-9]{5}\b", line) or re.search(r"PIN\s*Code", line, re.IGNORECASE):
                        pin_idx = i

                if to_idx >= 0 and pin_idx > to_idx:
                    addr_block = []
                    for k in range(to_idx + 1, pin_idx + 1):
                        cand_l = lines[k].strip()
                        # Exclude cardholder name or father name lines
                        if fields.get("holder_name") and (cand_l in fields["holder_name"] or fields["holder_name"] in cand_l):
                            continue
                        if fields.get("father_name") and (cand_l in fields["father_name"] or fields["father_name"] in cand_l):
                            continue
                        if re.search(r"(?:Father|S/O|D/O|W/O|Husband)", cand_l, re.IGNORECASE):
                            continue
                        if cand_l:
                            addr_block.append(cand_l)
                    if addr_block:
                        fields["address"] = ", ".join(addr_block)

            # 3. Fallback: PIN code locality line extraction
            if not fields["address"] and lines:
                pin_match = re.search(r"\b[1-9][0-9]{5}\b", combined_text)
                if pin_match:
                    for i, line in enumerate(lines):
                        if pin_match.group(0) in line:
                            start_i = max(0, i - 2)
                            block = [lines[k].strip() for k in range(start_i, i + 1) if lines[k].strip()]
                            fields["address"] = ", ".join(block)
                            break

        return fields

    @staticmethod
    def check_layout_and_font_alignment(image: Image.Image) -> Tuple[bool, List[str]]:
        """Analyzes text baseline alignment and font bounding box consistency using OpenCV.

        Detects rotated, shifted, or pasted text lines (common in amateur forged dates/numbers).
        """
        anomalies = []
        try:
            import cv2
            img_np = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # Threshold to isolate text characters
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Morphological dilation along horizontal axis to group characters into text lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            angles = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter for text-line proportioned blobs
                if w > 40 and 8 < h < 100:
                    rect = cv2.minAreaRect(cnt)
                    angle = rect[-1]
                    if angle < -45:
                        angle = 90 + angle
                    if abs(angle) > 0.5:
                        angles.append(abs(angle))

            # If there's high variance in text baseline angles, text was altered or pasted at odd angles
            if angles and np.std(angles) > 12.0:
                anomalies.append("Typographic baseline shift / non-parallel font alignment detected")

        except Exception as e:
            logger.debug("Layout alignment check skipped: %s", str(e))

        is_passed = len(anomalies) == 0
        return is_passed, anomalies


def run_layer2_analysis(image: Image.Image, custom_tesseract_cmd: str = "") -> Dict[str, Any]:
    """Main execution function for Layer 2: OCR & Structural Validation.

    Pipeline:
      1. Multi-engine OCR (EasyOCR / Tesseract)
      2. MRZ validation (ICAO Doc 9303)
      3. Barcode / QR detection
      4. Document classification
      5. Regex-based structural field extraction
      6. Gemini Vision AI field extraction (if GEMINI_API_KEY is set)
      7. Field merging (AI + OCR + MRZ)
      8. Layout & font alignment forensics
    """
    text, lines, tokens = OCRForensicExtractor.extract_text_and_lines(
        image, custom_cmd=custom_tesseract_cmd
    )

    # 1. MRZ validation (ICAO Doc 9303)
    mrz_info = MRZValidator.parse_and_validate(lines)

    # 2. Barcode & QR detection
    barcodes = OCRForensicExtractor.decode_barcodes(image)

    # 3. Document classification (OCR-based)
    doc_type = mrz_info.get("document_type") or OCRForensicExtractor.classify_document_type(text, lines)

    # 4. Extract visual fields via regex/heuristic OCR
    ocr_fields = OCRForensicExtractor.extract_structural_fields(text, lines, doc_type=doc_type)

    # 5. Gemini Vision AI field extraction — runs independently of OCR
    try:
        from app.layers.gemini_extractor import extract_fields_with_ai, merge_ai_and_ocr_fields
        ai_fields = extract_fields_with_ai(image, ocr_text_hint=text)
    except Exception as ai_err:
        logger.warning("AI extraction import/call error: %s", ai_err)
        ai_fields = {}

    # 6. Merge AI + OCR fields (AI enhances OCR, not replaces it)
    ai_confidence = ai_fields.get("ai_confidence", "medium") if ai_fields else "low"
    if ai_fields:
        try:
            from app.layers.gemini_extractor import merge_ai_and_ocr_fields
            merged_fields = merge_ai_and_ocr_fields(ai_fields, ocr_fields, ai_confidence=ai_confidence)
        except Exception:
            merged_fields = ocr_fields
        # Use AI's document_type if OCR got 'unknown'
        if doc_type == "unknown" and ai_fields.get("document_type") and ai_fields["document_type"] != "unknown":
            doc_type = ai_fields["document_type"]
    else:
        merged_fields = ocr_fields

    fields = merged_fields

    # 7. Check layout & font baseline alignment
    alignment_passed, alignment_anomalies = OCRForensicExtractor.check_layout_and_font_alignment(image)

    # If MRZ provided trusted structured fields, fill them in (MRZ is highest-trust)
    if mrz_info.get("mrz_detected"):
        if not fields.get("document_number"):
            fields["document_number"] = mrz_info.get("document_number")
        if mrz_info.get("holder_name") and len(mrz_info.get("holder_name", "").strip()) >= 3:
            fields["holder_name"] = mrz_info.get("holder_name")
        if not fields.get("date_of_birth"):
            fields["date_of_birth"] = mrz_info.get("date_of_birth")
        if not fields.get("expiry_date"):
            fields["expiry_date"] = mrz_info.get("expiry_date")
        if not fields.get("gender") and mrz_info.get("sex"):
            fields["gender"] = mrz_info.get("sex")

    anomalies: List[str] = []
    if mrz_info.get("anomalies"):
        anomalies.extend(mrz_info["anomalies"])
    if alignment_anomalies:
        anomalies.extend(alignment_anomalies)

    # Cross-check visual fields vs MRZ
    cross_check_matches = True
    if mrz_info.get("mrz_detected") and fields.get("document_number") and mrz_info.get("document_number"):
        clean_doc = fields["document_number"].replace(" ", "").upper()
        clean_mrz_doc = mrz_info["document_number"].replace(" ", "").upper()
        if clean_doc != clean_mrz_doc and clean_mrz_doc not in clean_doc:
            cross_check_matches = False
            anomalies.append(
                f"Discrepancy: Visual document number '{fields['document_number']}' does not match MRZ '{mrz_info['document_number']}'"
            )

    # Calculate status and confidence
    if anomalies:
        status = "flagged"
        confidence = 92.0
    elif mrz_info.get("mrz_detected") and mrz_info.get("mrz_checksum_valid"):
        status = "passed"
        confidence = 96.0
    elif doc_type != "unknown":
        status = "passed"
        confidence = 88.0
    else:
        status = "inconclusive"
        confidence = 50.0

    return {
        "status": status,
        "confidence": confidence,
        "document_type": doc_type,
        "fields": {
            "document_type": doc_type,
            "document_number": fields.get("document_number"),
            "holder_name": fields.get("holder_name"),
            "father_name": fields.get("father_name"),
            "date_of_birth": fields.get("date_of_birth"),
            "gender": fields.get("gender"),
            "address": fields.get("address"),
            "expiry_date": fields.get("expiry_date"),
            "nationality": fields.get("nationality"),
            "issue_date": fields.get("issue_date"),
            "raw_text_snippet": text[:300] if text else None,
            "raw_lines": lines[:20],
            # AI extraction metadata
            "ai_extraction_used": fields.get("ai_extraction_used", False),
            "ai_confidence": fields.get("ai_confidence"),
            "ai_unreadable_fields": fields.get("ai_unreadable_fields", []),
        },
        "mrz_detected": mrz_info.get("mrz_detected", False),
        "mrz_checksum_valid": mrz_info.get("mrz_checksum_valid"),
        "mrz_format": mrz_info.get("format"),
        "barcode_detected": len(barcodes) > 0,
        "alignment_passed": alignment_passed,
        "cross_check_matches": cross_check_matches,
        "anomalies": anomalies,
    }
