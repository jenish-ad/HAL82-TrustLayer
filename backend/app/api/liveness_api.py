# import base64
# import json
# import os
# import subprocess
# import sys
# from typing import Any
# from datetime import datetime
# from pathlib import Path
# from uuid import uuid4

# from fastapi import FastAPI, File, Form, HTTPException, UploadFile
# from fastapi.middleware.cors import CORSMiddleware

# ROOT_DIR = Path(__file__).resolve().parents[3]
# LIVENESS_SCRIPT = ROOT_DIR / "ml" / "liveness-service" / "liveness.py"
# OUTPUT_DIR = ROOT_DIR / "ml" / "liveness-service" / "extracted_faces"
# UPLOAD_DIR = ROOT_DIR / "ml" / "docservice" / "uploads"
# CROP_OUTPUT_DIR = ROOT_DIR / "ml" / "docservice" / "crops"

# DOC_SERVICE_DIR = ROOT_DIR / "ml" / "docservice"
# if str(DOC_SERVICE_DIR) not in sys.path:
#     sys.path.append(str(DOC_SERVICE_DIR))

# FACE_SERVICE_SRC_DIR = ROOT_DIR / "ml" / "face-service" / "src"
# if str(FACE_SERVICE_SRC_DIR) not in sys.path:
#     sys.path.append(str(FACE_SERVICE_SRC_DIR))

# from detect_crop import DocumentDetector
# from embed import get_embedding
# from match import compare_embeddings

# app = FastAPI(title="KYC Liveness API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/health")
# def health_check():
#     return {"status": "ok"}


# detector = DocumentDetector()


# def _save_upload(file: UploadFile, destination_dir: Path, prefix: str) -> Path:
#     destination_dir.mkdir(parents=True, exist_ok=True)
#     extension = Path(file.filename or "").suffix or ".jpg"
#     destination_path = destination_dir / f"{prefix}_{uuid4().hex}{extension}"

#     with destination_path.open("wb") as out_file:
#         out_file.write(file.file.read())

#     return destination_path


# def _pick_photo_crop(detections: list[dict[str, Any]]) -> str | None:
#     """Pick the citizenship photo crop path from detector outputs."""
#     for detection in detections:
#         class_name = str(detection.get("class_name", "")).lower()
#         crop_path = detection.get("crop_path")
#         if class_name == "photo" and crop_path:
#             return str(crop_path)

#     for detection in detections:
#         crop_path = str(detection.get("crop_path") or "")
#         if "photo" in Path(crop_path).stem.lower():
#             return crop_path

#     return None


# def _latest_liveness_selfie() -> Path | None:
#     candidate = OUTPUT_DIR / "face.jpg"
#     return candidate if candidate.exists() else None


# @app.post("/api/kyc/upload")
# def upload_kyc_documents(
#     full_name: str = Form(...),
#     date_of_birth: str = Form(...),
#     gender: str = Form(...),
#     citizenship_number: str = Form(...),
#     permanent_address: str = Form(...),
#     current_address: str = Form(...),
#     selfie_image: UploadFile = File(...),
#     document_front: UploadFile = File(...),
#     document_back: UploadFile = File(...),
# ):
#     del full_name, date_of_birth, gender, citizenship_number
#     del permanent_address, current_address, selfie_image, document_back

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     front_image_path = _save_upload(
#         file=document_front,
#         destination_dir=UPLOAD_DIR,
#         prefix=f"front_{timestamp}",
#     )

#     crop_dir = CROP_OUTPUT_DIR / timestamp
#     detections = detector.detect_and_crop(
#         image_path=str(front_image_path),
#         output_dir=str(crop_dir),
#     )

#     if not detections:
#         raise HTTPException(
#             status_code=400,
#             detail="No front-side citizenship regions were detected in the uploaded image.",
#         )

#     doc_face_crop_path = _pick_photo_crop(detections)
#     if not doc_face_crop_path:
#         raise HTTPException(
#             status_code=400,
#             detail="Could not find citizenship photo crop from document detection results.",
#         )

#     liveness_selfie_path = _latest_liveness_selfie()
#     if not liveness_selfie_path:
#         raise HTTPException(
#             status_code=400,
#             detail="Live selfie image is missing. Please run active liveness first.",
#         )

#     try:
#         emb_doc = get_embedding(doc_face_crop_path)
#         emb_selfie = get_embedding(str(liveness_selfie_path))
#         similarity_result = compare_embeddings(emb_doc, emb_selfie)
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc)) from exc
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=f"Face similarity failed: {exc}") from exc

#     print(
#         "[FACE-SIMILARITY] "
#         f"doc={doc_face_crop_path} "
#         f"selfie={liveness_selfie_path} "
#         f"score={similarity_result['similarity']:.6f} "
#         f"match={similarity_result['match']}"
#     )

#     return {
#         "status": "processed",
#         "message": "Citizenship front image processed and crops saved.",
#         "uploaded_front_image": str(front_image_path),
#         "crop_directory": str(crop_dir),
#         "detections": detections,
#         "doc_face_crop": doc_face_crop_path,
#         "liveness_selfie": str(liveness_selfie_path),
#         "face_similarity": {
#             "similarity_score": float(similarity_result["similarity"]),
#             "match": bool(similarity_result["match"]),
#             "threshold_used": float(similarity_result["threshold_used"]),
#         },
#     }


# @app.post("/api/liveness/run")
# def run_liveness():
#     if not LIVENESS_SCRIPT.exists():
#         raise HTTPException(status_code=500, detail="ml/liveness-service/liveness.py not found")

#     cmd = [
#         sys.executable,
#         str(LIVENESS_SCRIPT),
#         "--json",
#         "--output-dir",
#         str(OUTPUT_DIR),
#     ]

#     completed = subprocess.run(
#         cmd,
#         cwd=str(ROOT_DIR),
#         capture_output=True,
#         text=True,
#         check=False,
#     )

#     stdout = completed.stdout.strip().splitlines()
#     result_line = stdout[-1] if stdout else "{}"

#     try:
#         result = json.loads(result_line)
#     except json.JSONDecodeError as exc:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Invalid liveness response: {result_line}",
#         ) from exc

#     image_path = result.get("image_path")
#     image_data_url = None

#     if image_path and os.path.exists(image_path):
#         with open(image_path, "rb") as image_file:
#             encoded = base64.b64encode(image_file.read()).decode("utf-8")
#         image_data_url = f"data:image/jpeg;base64,{encoded}"

#     return {
#         "passed": bool(result.get("passed")),
#         "message": result.get("message", "Liveness failed"),
#         "image": image_data_url,
#         "return_code": completed.returncode,
#         "stderr": completed.stderr.strip(),
#     }


import base64
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[3]
LIVENESS_SCRIPT = ROOT_DIR / "ml" / "liveness-service" / "liveness.py"
OUTPUT_DIR = ROOT_DIR / "ml" / "liveness-service" / "extracted_faces"
UPLOAD_DIR = ROOT_DIR / "ml" / "docservice" / "uploads"
CROP_OUTPUT_DIR = ROOT_DIR / "ml" / "docservice" / "crops"

DOC_SERVICE_DIR = ROOT_DIR / "ml" / "docservice"
if str(DOC_SERVICE_DIR) not in sys.path:
    sys.path.append(str(DOC_SERVICE_DIR))

OCR_SERVICE_SRC_DIR = ROOT_DIR / "ml" / "ocr-service" / "src"
if str(OCR_SERVICE_SRC_DIR) not in sys.path:
    sys.path.append(str(OCR_SERVICE_SRC_DIR))

FACE_SERVICE_SRC_DIR = ROOT_DIR / "ml" / "face-service" / "src"
if str(FACE_SERVICE_SRC_DIR) not in sys.path:
    sys.path.append(str(FACE_SERVICE_SRC_DIR))

from detect_crop import DocumentDetector
from parse_fields import parse_back

# ─────────────────────────────────────────────────────────────
# OPTIONAL FACE MATCH MODULES (DON'T BREAK API IF MISSING)
# ─────────────────────────────────────────────────────────────
try:
    from embed import get_embedding
    from match import compare_embeddings
except Exception as e:
    print(f"[FACE-MODULES] WARNING: embed/match not available -> {e}")
    get_embedding = None
    compare_embeddings = None

print("[FACE-MODULES] get_embedding:", "OK" if get_embedding else "MISSING")
print("[FACE-MODULES] compare_embeddings:", "OK" if compare_embeddings else "MISSING")

app = FastAPI(title="KYC Liveness API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


detector = DocumentDetector()


def _save_upload(file: UploadFile, destination_dir: Path, prefix: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(file.filename or "").suffix or ".jpg"
    destination_path = destination_dir / f"{prefix}_{uuid4().hex}{extension}"

    with destination_path.open("wb") as out_file:
        out_file.write(file.file.read())

    return destination_path


def _pick_photo_crop(detections: list[dict[str, Any]]) -> str | None:
    """Pick the citizenship photo crop path from detector outputs."""
    for detection in detections:
        class_name = str(detection.get("class_name", "")).lower()
        crop_path = detection.get("crop_path")
        if class_name == "photo" and crop_path:
            return str(crop_path)

    for detection in detections:
        crop_path = str(detection.get("crop_path") or "")
        if "photo" in Path(crop_path).stem.lower():
            return crop_path

    return None


def _run_face_similarity(doc_face_crop_path: str, selfie_path: str) -> dict[str, Any] | None:
    """
    Returns normalized keys:
      - similarity (float)
      - match (bool)
      - threshold_used (float)
    """
    if not get_embedding or not compare_embeddings:
        return None

    emb_doc = get_embedding(doc_face_crop_path)
    emb_selfie = get_embedding(selfie_path)

    if emb_doc is None or emb_selfie is None:
        raise ValueError("Face embedding could not be generated (no face detected or model failed).")

    similarity_result = compare_embeddings(emb_doc, emb_selfie)

    return {
        "similarity": float(similarity_result.get("similarity", 0.0)),
        "match": bool(similarity_result.get("match", False)),
        "threshold_used": float(similarity_result.get("threshold_used", 0.0)),
    }


@app.post("/api/kyc/upload")
def upload_kyc_documents(
    full_name: str = Form(...),
    date_of_birth: str = Form(...),
    gender: str = Form(...),
    citizenship_number: str = Form(...),
    permanent_address: str = Form(...),
    current_address: str = Form(...),
    selfie_image: UploadFile = File(...),
    document_front: UploadFile = File(...),
    document_back: UploadFile = File(...),
):
    # (kept but unused in this endpoint flow)
    del full_name, date_of_birth, gender, citizenship_number
    del permanent_address, current_address

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    front_image_path = _save_upload(
        file=document_front,
        destination_dir=UPLOAD_DIR,
        prefix=f"front_{timestamp}",
    )
    back_image_path = _save_upload(
        file=document_back,
        destination_dir=UPLOAD_DIR,
        prefix=f"back_{timestamp}",
    )
    selfie_path = _save_upload(
        file=selfie_image,
        destination_dir=UPLOAD_DIR,
        prefix=f"selfie_{timestamp}",
    )

    front_crop_dir = CROP_OUTPUT_DIR / timestamp / "front"
    back_crop_dir = CROP_OUTPUT_DIR / timestamp / "back"

    front_detections = detector.detect_front(
        image_path=str(front_image_path),
        output_dir=str(front_crop_dir),
    )
    back_detections = detector.detect_back(
        image_path=str(back_image_path),
        output_dir=str(back_crop_dir),
    )

    if not front_detections:
        raise HTTPException(
            status_code=400,
            detail="No front-side citizenship regions were detected in the uploaded image.",
        )

    if not back_detections:
        raise HTTPException(
            status_code=400,
            detail="No back-side citizenship fields were detected in the uploaded image.",
        )

    parsed_fields = parse_back(
        image_path=str(back_image_path),
        detector=detector,
        output_dir=str(back_crop_dir),
    )

    doc_face_crop_path = _pick_photo_crop(front_detections)

    face_similarity: dict[str, Any] | None = None
    if doc_face_crop_path:
        try:
            face_similarity = _run_face_similarity(doc_face_crop_path, str(selfie_path))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            # Do not fail document OCR flow because of optional face matcher issues.
            face_similarity = {"error": f"Face similarity failed: {exc}"}
    else:
        face_similarity = {"error": "No citizenship photo crop found in front detections."}

    # ✅ SAFE TERMINAL LOGGING (won't crash)
    if face_similarity is None:
        print("[FACE-SIMILARITY] Skipped: embed/match modules not available.")
    elif isinstance(face_similarity, dict) and "error" in face_similarity:
        print(f"[FACE-SIMILARITY] Failed: {face_similarity['error']}")
    else:
        print(
            "[FACE-SIMILARITY] "
            f"doc={doc_face_crop_path} "
            f"selfie={selfie_path} "
            f"score={face_similarity['similarity']:.6f} "
            f"match={face_similarity['match']} "
            f"threshold={face_similarity['threshold_used']}"
        )

    return {
        "status": "processed",
        "message": "Citizenship front/back processed and OCR extracted.",
        "uploaded_front_image": str(front_image_path),
        "uploaded_back_image": str(back_image_path),
        "uploaded_selfie_image": str(selfie_path),
        "front_crop_directory": str(front_crop_dir),
        "back_crop_directory": str(back_crop_dir),
        "front_detections": front_detections,
        "back_detections": back_detections,
        "parsed_fields": parsed_fields,
        "doc_face_crop": doc_face_crop_path,
        "face_similarity": face_similarity,
    }


@app.post("/api/liveness/run")
def run_liveness():
    if not LIVENESS_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="ml/liveness-service/liveness.py not found")

    cmd = [
        sys.executable,
        str(LIVENESS_SCRIPT),
        "--json",
        "--output-dir",
        str(OUTPUT_DIR),
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = completed.stdout.strip().splitlines()
    result_line = stdout[-1] if stdout else "{}"

    try:
        result = json.loads(result_line)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid liveness response: {result_line}",
        ) from exc

    image_path = result.get("image_path")
    image_data_url = None

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{encoded}"

    return {
        "passed": bool(result.get("passed")),
        "message": result.get("message", "Liveness failed"),
        "image": image_data_url,
        "return_code": completed.returncode,
        "stderr": completed.stderr.strip(),
    }