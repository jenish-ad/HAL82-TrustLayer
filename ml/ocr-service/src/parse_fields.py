

import cv2
from ocr import extract_text
from utils import KycCleaner

CLASS_NAME_MAP = {
    "citisenship-number":  "citisenship-number",
    "full-name":           "full-name",
    "DOB":                 "DOB",
    "permanent-address":   "permanent-address",
    "officer-signature":   "officer-signature",
}

LOW_CONF_OK = {"officer_signature"}

MIN_DETECTION_CONF = 0.5


# ── 2. CORE PARSER ────────────────────────────────────────────────────────────

def parse_fields(detections: list[dict]) -> dict:
   
    cleaner = KycCleaner()
    result = {}
    unrecognised = []

    for det in detections:
        class_name = det.get("class_name", "")
        det_conf   = det.get("confidence", 0.0)
        crop_path  = det.get("crop_path", "")

        # ── Skip if class not mapped ──
        if class_name not in CLASS_NAME_MAP:
            unrecognised.append(class_name)
            continue

        field = CLASS_NAME_MAP[class_name]
        if field is None:
            continue 

       
        if det_conf < MIN_DETECTION_CONF and field not in LOW_CONF_OK:
            result[field] = {
                "raw":        "",
                "cleaned":    "[NOT_DETECTED]",
                "confidence": det_conf,
            }
            continue

        # ── Load the saved crop from disk ──
        crop_img = cv2.imread(crop_path)
        if crop_img is None:
            result[field] = {
                "raw":        "",
                "cleaned":    "[NOT_DETECTED]",
                "confidence": 0.0,
            }
            continue

        # ── Run OCR ──
        raw_text, ocr_conf = extract_text(crop_img, field_type=field)

        # ── Clean / validate ──
        cleaned = cleaner.clean(field, raw_text, conf=ocr_conf)

        result[field] = {
            "raw":        raw_text,
            "cleaned":    cleaned,
            "confidence": round(ocr_conf, 3),
        }

    # ── Fill in any fields that were never detected ──
    for field in set(CLASS_NAME_MAP.values()):
        if field is None:
            continue
        if field not in result:
            cleaned = cleaner.clean(field, None)
            result[field] = {
                "raw":        "",
                "cleaned":    cleaned,
                "confidence": 0.0,
            }

    result["_unrecognised"] = unrecognised
    return result


# ── 3. CONVENIENCE WRAPPER ────────────────────────────────────────────────────

def parse_back(image_path: str, detector, output_dir: str = "crops/back") -> dict:
    
    detections = detector.detect_back(image_path, output_dir=output_dir)
    return parse_fields(detections)


def parse_front(image_path: str, detector, output_dir: str = "crops/front") -> dict:
   
    detections = detector.detect_front(image_path, output_dir=output_dir)
    return {
        "detections": detections,
        "note": "Front side — no KYC text fields extracted",
    }