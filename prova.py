from ultralytics import YOLO
import cv2

# ─── CONFIGURAZIONE ───────────────────────────────────────────────────
MODE = "detection"  # "detection" → solo bounding box
                       # "segmentation" → solo maschera

IMAGE_PATH = "datasetAggiornato/valid/images/1797-60-120_color_jpg.rf.b2032af5a659001317a397fa42d14492.jpg"
OUTPUT_PATH = "output.jpg"

MODEL_PATH = "runs/segment/oldDatasetTrain/large11DataOld-BEST/weights/best.pt"
# ─────────────────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Immagine non trovata: {IMAGE_PATH}")

results = model.predict(frame, conf=0.40, verbose=False)

if MODE == "detection":
    annotated_frame = results[0].plot(masks=False, boxes=True)
elif MODE == "segmentation":
    annotated_frame = results[0].plot(masks=True, boxes=False)
else:
    raise ValueError(f"MODE non valido: '{MODE}'. Usa 'detection' o 'segmentation'.")

cv2.imwrite(OUTPUT_PATH, annotated_frame)
print(f"Risultato salvato in: {OUTPUT_PATH}")
