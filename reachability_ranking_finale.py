"""
Algoritmo di Reachability Ranking per la selezione dei target di raccolta.
Seleziona i 3 pomodori più raggiungibili in base a: area, circolarità, centralità e maturità.
Output: immagine con maschere colorate solo sui top 3 target.
"""

import cv2
import numpy as np
import math
import os
from ultralytics import YOLO

# ── Configurazione ──────────────────────────────────────────────────
MODEL_PATH   = "runs/segment/oldDatasetTrain/large11DataOld-BEST/weights/best.pt"
IMAGE_DIR    = "datasetAggiornato/test/images"
OUTPUT_DIR   = "output_reachability"
TARGET_CLASS = 3          # Classe "Tomato"
CONF         = 0.3        # Soglia di confidenza YOLO
IMGSZ        = 800        # Risoluzione di inferenza
AREA_MIN     = 1500       # Area minima in pixel (filtra frutti troppo lontani)
MAX_TARGETS  = 3          # Numero di target da selezionare

# Pesi della funzione di costo (somma = 1.0)
W_AREA        = 0.35
W_CIRCULARITY = 0.25
W_CENTRALITY  = 0.15
W_MATURITY    = 0.25

# Colori di visualizzazione per i rank (BGR)
RANK_COLORS = {
    1: (0, 255, 0),    # Verde  — Rank 1 (miglior target)
    2: (0, 200, 255),  # Arancio — Rank 2
    3: (0, 100, 255),  # Rosso-arancio — Rank 3
}


def maturity_score(image_bgr, binary_mask):
    """
    Calcola il grado di maturità del pomodoro analizzando la dominanza
    cromatica nella regione della maschera nello spazio HSV.
    
    Pomodori maturi (rossi):  H ∈ [0, 15] ∪ [165, 180], S > 50, V > 50
    Pomodori acerbi (verdi):  H ∈ [35, 85],  S > 40, V > 40
    
    Restituisce un valore in [0, 1]: 1.0 = completamente maturo, 0.0 = completamente acerbo.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Maschere cromatiche
    mask_red1 = cv2.inRange(hsv, (0, 50, 50),   (15, 255, 255))
    mask_red2 = cv2.inRange(hsv, (165, 50, 50),  (180, 255, 255))
    mask_red  = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, (35, 40, 40),  (85, 255, 255))
    
    # Intersezione con la maschera del pomodoro
    red_pixels   = int(np.sum((mask_red > 0) & (binary_mask > 0)))
    green_pixels = int(np.sum((mask_green > 0) & (binary_mask > 0)))
    total = red_pixels + green_pixels
    
    if total == 0:
        return 0.5  # Indeterminato (es. pomodoro arancione in transizione)
    
    return red_pixels / total


def compute_reachability(mask_binary, image_bgr, img_h, img_w):
    """
    Calcola lo score di raggiungibilità per una singola istanza.
    Restituisce (score, metriche_dict) oppure None se l'istanza non supera i filtri.
    """
    area = int(np.sum(mask_binary > 0))
    if area < AREA_MIN:
        return None
    
    # Circolarità: C = 4π·A / P²  (1.0 = cerchio perfetto)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    perimeter = cv2.arcLength(contours[0], True)
    circularity = min(1.0, (4 * math.pi * area) / (perimeter ** 2)) if perimeter > 0 else 0
    
    # Centralità: distanza normalizzata del centroide dal centro ottico
    coords = np.where(mask_binary > 0)
    cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
    max_dist = math.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
    dist = math.sqrt((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2)
    centrality = 1.0 - (dist / max_dist)
    
    # Area normalizzata (proxy della vicinanza)
    area_norm = min(1.0, area / (img_w * img_h * 0.15))
    
    # Maturità cromatica
    maturity = maturity_score(image_bgr, mask_binary)
    
    # Score composito
    score = (W_AREA * area_norm
           + W_CIRCULARITY * circularity
           + W_CENTRALITY * centrality
           + W_MATURITY * maturity)
    
    return {
        "score": score,
        "area": area,
        "area_norm": area_norm,
        "circularity": circularity,
        "centrality": centrality,
        "maturity": maturity,
        "centroid": (cx, cy),
        "mask": mask_binary,
    }


def overlay_mask(image, mask, color, alpha=0.45):
    """Sovrappone una maschera colorata semi-trasparente sull'immagine."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_rank_label(image, centroid, rank, score, color):
    """Disegna etichetta di ranking sul centroide."""
    cx, cy = centroid
    # Cerchio bianco sul centroide
    cv2.circle(image, (cx, cy), 6, (255, 255, 255), -1)
    cv2.circle(image, (cx, cy), 6, color, 2)
    # Label
    label = f"#{rank}  {score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    lx, ly = cx - tw // 2, cy - 18
    # Sfondo scuro per leggibilità
    cv2.rectangle(image, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), (0, 0, 0), -1)
    cv2.putText(image, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def process_folder(model_path=MODEL_PATH, image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR):
    """Processa tutte le immagini nella cartella e salva i risultati."""
    
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(extensions)])
    
    if not images:
        print(f"Nessuna immagine trovata in {image_dir}")
        return
    
    print(f"Elaborazione di {len(images)} immagini...")
    
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        results = model.predict(img_path, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        img = results.orig_img.copy()
        h, w = img.shape[:2]
        
        candidates = []
        
        if results.masks is not None:
            for i, mask_tensor in enumerate(results.masks.data):
                if int(results.boxes.cls[i]) != TARGET_CLASS:
                    continue
                
                # Ridimensiona maschera alla risoluzione originale
                m = mask_tensor.cpu().numpy()
                m = cv2.resize(m, (w, h))
                binary = (m > 0.5).astype(np.uint8) * 255
                
                result = compute_reachability(binary, img, h, w)
                if result is not None:
                    candidates.append(result)
        
        # Ranking per score decrescente → top 3
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:MAX_TARGETS]
        
        # Visualizzazione: maschera colorata + label solo sui top target
        output = img.copy()
        for rank, det in enumerate(top, 1):
            color = RANK_COLORS[rank]
            output = overlay_mask(output, det["mask"], color)
            draw_rank_label(output, det["centroid"], rank, det["score"], color)
        
        save_path = os.path.join(output_dir, f"ranked_{img_name}")
        cv2.imwrite(save_path, output)
    
    print(f"Completato. Risultati salvati in: {output_dir}/")


if __name__ == "__main__":
    process_folder()