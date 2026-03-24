import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

# Configurazione aggiornata per 448x448
MODEL_PATH = "runs/segment/m26Finale/weights/best.pt"
IMAGE_DIR = "datasetAggiornato/test/images"
TARGET_CLASS = 3  
AREA_THRESHOLD = 2000  # Soglia più sensata per 448x448
MAX_TARGETS = 3       # Mostriamo solo i Top 3
OUTPUT_DIR = "visualizzazione_raggiungibilita"

def calculate_reachability():
    # ... (caricamento modello e cartelle uguale)
    if not os.path.exists(MODEL_PATH):
        print(f"Errore: Modello non trovato in {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))][:10] # Aumentiamo a 10 per vedere più casi
    
    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        results = model.predict(img_path, conf=0.3, imgsz=448, verbose=False)[0]
        img_draw = results.orig_img.copy()
        
        detections = []
        if results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                if int(results.boxes.cls[i]) != TARGET_CLASS:
                    continue
                
                m = mask.cpu().numpy()
                m = cv2.resize(m, (img_draw.shape[1], img_draw.shape[0]))
                area = int(np.sum(m > 0.5))
                
                coords = np.where(m > 0.5)
                if len(coords[0]) > 0:
                    cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
                    detections.append({
                        'area': area,
                        'center': (cx, cy),
                        'box': results.boxes.xyxy[i].cpu().numpy()
                    })

        # Ranking per area (descrescente)
        detections.sort(key=lambda x: x['area'], reverse=True)
        
        # Prendiamo solo i Top 3
        top_targets = detections[:MAX_TARGETS]
        
        for rank, det in enumerate(top_targets, 1):
            is_valid = det['area'] > AREA_THRESHOLD
            color = (0, 255, 0) if is_valid else (0, 0, 255) # Verde se OK, Rosso se troppo piccolo
            
            x1, y1, x2, y2 = det['box'].astype(int)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # Label specifica
            status = "OK" if is_valid else "PICCOLO"
            label = f"Top {rank}: {status} ({det['area']}px)"
            cv2.putText(img_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Numero ranking al centro
            cv2.putText(img_draw, str(rank), (det['center'][0]-10, det['center'][1]+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        save_path = os.path.join(OUTPUT_DIR, f"top3_{img_name}")
        cv2.imwrite(save_path, img_draw)
        print(f"Salvata visualizzazione: {save_path} (Targets mostrati: {len(top_targets)})")

if __name__ == "__main__":
    calculate_reachability()
