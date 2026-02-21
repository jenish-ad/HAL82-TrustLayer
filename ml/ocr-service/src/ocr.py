import easyocr
import numpy as np
import cv2

reader = easyocr.Reader(['en'], gpu=False)

def preprocess_image(crop_img, field_type=None):
    # Step 1: Upscale only
    h, w = crop_img.shape[:2]
    img = cv2.resize(crop_img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    # Step 2: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Light denoise only (reduced h=5 from h=10)
    denoised = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=5, searchWindowSize=15)

    if field_type == "citisenship-number":
        # Light CLAHE only — no sharpening kernel, no Otsu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(denoised)

    elif field_type == "DOB":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(denoised)

    elif field_type == "full-name":
        # Very gentle adaptive threshold
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        processed = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,   # larger block = less aggressive
            C=15
        )

    elif field_type == "permanent-address":
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        processed = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,
            C=12
        )

    elif field_type == "officer-signature":
        _, processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    else:
        # Minimal fallback — just CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        processed = clahe.apply(denoised)

    # Light padding
    processed = cv2.copyMakeBorder(processed, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    return processed


def extract_text(crop_img, field_type=None):
    if crop_img is None or crop_img.size == 0:
        return "", 0.0

    try:
        # Run OCR on BOTH original and preprocessed, pick better result
        h, w = crop_img.shape[:2]
        original_upscaled = cv2.resize(crop_img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        preprocessed = preprocess_image(crop_img, field_type)

        results_original     = reader.readtext(original_upscaled, paragraph=False, decoder='beamsearch')
        results_preprocessed = reader.readtext(preprocessed,       paragraph=False, decoder='beamsearch')

        def parse_results(results):
            text_list, confidences = [], []
            for res in results:
                if not res or len(res) < 3:
                    continue
                detected_text = str(res[1]).strip()
                confidence    = float(res[2])
                if field_type == "citisenship-number":
                    detected_text = detected_text.replace(" ", "").replace(".", "")
                if detected_text:
                    text_list.append(detected_text)
                    confidences.append(confidence)
            full_text = " ".join(t for t in text_list if len(t) > 1).strip()
            avg_conf  = sum(confidences) / len(confidences) if confidences else 0.0
            return full_text, avg_conf

        text_orig, conf_orig = parse_results(results_original)
        text_pre,  conf_pre  = parse_results(results_preprocessed)

        # Pick whichever gave higher confidence
        if conf_orig >= conf_pre:
            print(f"[{field_type}] Using ORIGINAL — conf={conf_orig:.3f} | text='{text_orig}'")
            return text_orig, conf_orig
        else:
            print(f"[{field_type}] Using PREPROCESSED — conf={conf_pre:.3f} | text='{text_pre}'")
            return text_pre, conf_pre

    except Exception as e:
        print(f"Error in extract_text for {field_type}: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0