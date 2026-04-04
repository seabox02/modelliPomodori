import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import math

# Configurazione per YOLO11-Large (quello identificato come Best)
MODEL_PATH = "runs/segment/newDatasetTrain/extralarge26Best/weights/best.pt" # Usiamo l'XL o l'11L come discusso
IMAGE_DIR = "datasetAggiornato/test/images"
TARGET_CLASS = 3  # Tomato
OUTPUT_DIR = "visualizzazione_raggiungibilita"

# Nuovi Pesi per favorire il Primo Piano
W_AREA = 0.5         # Più grande = Più vicino (Primo Piano)
W_CIRCULARITY = 0.3  # Forma sferica
W_CENTRALITY = 0.2   # Posizione nel frame

# Soglie aggiornate
AREA_MIN_THRESHOLD = 1500 # Ignoriamo i pomodori troppo piccoli/lontani
SCORE_THRESHOLD = 0.5
MAX_TARGETS = 3

def calculate_iou(box1, box2):
    """Calcola l'Intersection over Union tra due bounding box."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_graspability():
    if not os.path.exists(MODEL_PATH):
        print(f"Errore: Modello non trovato in {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))][:15]
    
    imgsz = 800
    
    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        results = model.predict(img_path, conf=0.3, imgsz=imgsz, verbose=False)[0]
        img_draw = results.orig_img.copy()
        h, w = img_draw.shape[:2]
        img_center = (w // 2, h // 2)
        max_dist = math.sqrt((w//2)**2 + (h//2)**2)
        
        detections = []
        if results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                if int(results.boxes.cls[i]) != TARGET_CLASS:
                    continue
                
                m = mask.cpu().numpy()
                m = cv2.resize(m, (w, h))
                binary_mask = (m > 0.5).astype(np.uint8) * 255
                
                area = int(np.sum(binary_mask > 0))
                if area < AREA_MIN_THRESHOLD: continue 
                
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                cnt = contours[0]
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                circularity = min(1.0, circularity)
                
                coords = np.where(binary_mask > 0)
                cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
                dist_from_center = math.sqrt((cx - img_center[0])**2 + (cy - img_center[1])**2)
                centrality = 1.0 - (dist_from_center / max_dist)
                
                area_norm = min(1.0, area / (w * h * 0.15)) # Normalizzazione su 15% del frame
                
                score = (W_AREA * area_norm) + (W_CIRCULARITY * circularity) + (W_CENTRALITY * centrality)
                
                detections.append({
                    'score': score,
                    'area': area,
                    'circularity': circularity,
                    'centrality': centrality,
                    'center': (cx, cy),
                    'box': results.boxes.xyxy[i].cpu().numpy(),
                    'id': i
                })

        # --- LOGICA DI PENALITÀ PER OCCLUSIONE (Chi sta dietro perde punti) ---
        for i in range(len(detections)):
            for j in range(len(detections)):
                if i == j: continue
                
                iou = calculate_iou(detections[i]['box'], detections[j]['box'])
                if iou > 0.2: # Se c'è una sovrapposizione significativa
                    # Se il pomodoro i è più piccolo del pomodoro j, i riceve una penalità
                    # perché è probabile che sia quello occluso o in secondo piano.
                    if detections[i]['area'] < detections[j]['area']:
                        detections[i]['score'] *= 0.7 # Penalità del 30%
                        # print(f"Penalità applicata a pomodoro {i} causa sovrapposizione con {j}")

        # Ranking Finale
        detections.sort(key=lambda x: x['score'], reverse=True)
        top_targets = detections[:MAX_TARGETS]
        
        for rank, det in enumerate(top_targets, 1):
            is_optimal = det['score'] > SCORE_THRESHOLD
            color = (0, 255, 0) if is_optimal else (0, 165, 255)
            
            x1, y1, x2, y2 = det['box'].astype(int)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            label_top = f"Rank {rank}: Score {det['score']:.2f}"
            label_metrics = f"C:{det['circularity']:.2f} A:{det['area']}"
            
            cv2.putText(img_draw, label_top, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(img_draw, label_metrics, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.circle(img_draw, det['center'], 5, (255, 255, 255), -1)
            cv2.putText(img_draw, str(rank), (det['center'][0]-10, det['center'][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        save_path = os.path.join(OUTPUT_DIR, f"ranked_{img_name}")
        cv2.imwrite(save_path, img_draw)

if __name__ == "__main__":
    calculate_graspability()
