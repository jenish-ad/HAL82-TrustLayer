import os
import cv2
from ultralytics import YOLO
import shutil 

class DocumentDetector:
    def __init__(self, front_model_path=None, back_model_path=None):
        base_path = os.path.dirname(os.path.abspath(__file__))

        if front_model_path is None:
            front_model_path = os.path.join(base_path, "weights", "frontbest.pt")
        if back_model_path is None:
            back_model_path = os.path.join(base_path, "weights", "backbest.pt")

        self.front_model = YOLO(front_model_path)
        self.back_model  = YOLO(back_model_path)

        self.front_target_classes = [2, 4]  # logo, photo only

    def _run_detection(self, model, image_path, output_dir, target_classes=None):
        img = cv2.imread(image_path)
        if img is None:
            return []

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        predict_kwargs = dict(source=image_path, conf=0.5)
        if target_classes is not None:
            predict_kwargs["classes"] = target_classes

        results = model.predict(**predict_kwargs)
       

        detections = []
        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf        = float(box.conf[0])
                class_id    = int(box.cls[0])
                class_name  = model.names[class_id]

                # In _run_detection, add slight bbox padding before cropping
                pad = 10
                h_img, w_img = img.shape[:2]
                x1 = max(0, x1 - pad)
                y1 = max(0, y1)
                x2 = min(w_img, x2 + pad)
                y2 = min(h_img, y2)

                cropped_img = img[y1:y2, x1:x2]
                crop_filename = f"crop_{class_name}_{i}.jpg"
                crop_path    = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, cropped_img)

                detections.append({
                    "class_id":   class_id,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox":       [x1, y1, x2, y2],
                    "crop_path":  crop_path,
                })

        return detections

    def detect_front(self, image_path, output_dir="crops/front"):
        return self._run_detection(
            self.front_model,
            image_path,
            output_dir,
            target_classes=self.front_target_classes,  # only class 2 & 4
        )

    def detect_back(self, image_path, output_dir="crops/back"):
        return self._run_detection(
            self.back_model,
            image_path,
            output_dir,
            target_classes=None,  # detect ALL fields
        )
    

    #detect_crop.py