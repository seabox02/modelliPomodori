"""
Algoritmo di Reachability Ranking per la selezione dei target di raccolta.
Architettura a due stadi:
  1. Ranking geometrico (area, circolarità, centralità)
  2. Filtro di maturità cromatica (HSV)
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
TARGET_CLASS = 3          # la classe tomato
CONF         = 0.3        # soglia di confidenza scelta
IMGSZ        = 800        
AREA_MIN     = 1500       # area minima in pixel, scartando quelli troppo piccoli 
MAX_TARGETS  = 3          # numero di target che verranno selezionati (top 3)

# pesi del ranking geometrico
W_AREA        = 0.45
W_CIRCULARITY = 0.35
W_CENTRALITY  = 0.20

# soglia di maturità per il filtro cromatico: 0.0 = acerbo, 1.0 = maturo
MATURITY_THRESHOLD = 0.4

# sottoinsieme di candidati geometrici da cui filtrare (> MAX_TARGETS)
GEOMETRIC_POOL = 8

# colori di visualizzazione per i rank (BGR)
RANK_COLORS = {
    1: (0, 255, 0),    # Verde  — Rank 1 (miglior target)
    2: (0, 200, 255),  # Arancio — Rank 2
    3: (0, 100, 255),  # Rosso-arancio — Rank 3
}


# fase 1: criterio geometrico

def compute_geometric_score(mask_binary, img_h, img_w):
    """
    Calcola lo score di raggiungibilità geometrica per una singola istanza.
    Restituisce il dizionario con metriche e score, oppure None se filtrata.
    """
    area = int(np.sum(mask_binary > 0))
    if area < AREA_MIN:
        return None

    # circolarità: 
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    perimeter = cv2.arcLength(contours[0], True)
    circularity = min(1.0, (4 * math.pi * area) / (perimeter ** 2)) if perimeter > 0 else 0

    # centralità: distanza normalizzata del centroide dal centro ottico
    coords = np.where(mask_binary > 0)
    cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
    max_dist = math.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
    dist = math.sqrt((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2)
    centrality = 1.0 - (dist / max_dist)

    # area normalizzata, indice di vicinanza
    area_norm = min(1.0, area / (img_w * img_h * 0.15))

    # score geometrico 
    score = (W_AREA * area_norm
           + W_CIRCULARITY * circularity
           + W_CENTRALITY * centrality)

    return {
        "geo_score": score,
        "area": area,
        "area_norm": area_norm,
        "circularity": circularity,
        "centrality": centrality,
        "centroid": (cx, cy),
        "mask": mask_binary,
    }


# fase 2: filtro di maturità cromatica 

def compute_maturity(image_bgr, binary_mask):
    """
    Stima la maturità del pomodoro analizzando la dominanza cromatica
    nella regione della maschera, nello spazio HSV.

    Pomodori maturi (rossi):  H ∈ [0, 15] ∪ [165, 180], S > 50, V > 50
    Pomodori acerbi (verdi):  H ∈ [35, 85],  S > 40, V > 40

    Restituisce un valore in [0, 1]: 1.0 = maturo, 0.0 = acerbo.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # maschere cromatiche
    mask_red1  = cv2.inRange(hsv, (0, 50, 50),   (15, 255, 255))
    mask_red2  = cv2.inRange(hsv, (165, 50, 50),  (180, 255, 255))
    mask_red   = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, (35, 40, 40),   (85, 255, 255))

    # intersezione con la maschera del pomodoro
    red_pixels   = int(np.sum((mask_red > 0) & (binary_mask > 0)))
    green_pixels = int(np.sum((mask_green > 0) & (binary_mask > 0)))
    total = red_pixels + green_pixels

    if total == 0:
        return 0.5  # indeterminato (es. pomodoro arancione in transizione)

    return red_pixels / total

# visualizzazione 

def overlay_mask(image, mask, color, alpha=0.45):
    """sovrappone una maschera colorata semi-trasparente sull'immagine."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_rank_marker(image, centroid, rank, color):
    """disegna solo il numero di rank sul centroide."""
    cx, cy = centroid
    cv2.circle(image, (cx, cy), 12, color, -1)
    cv2.circle(image, (cx, cy), 12, (255, 255, 255), 2)
    cv2.putText(image, str(rank), (cx - 5, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_legend(image, targets):
    """disegna una legenda in basso a sinistra con le metriche di ogni target."""
    line_h = 28
    padding = 10
    box_h = padding * 2 + line_h * len(targets)
    box_w = 320
    h, w = image.shape[:2]

    # posizione: basso a sinistra
    x0, y0 = 10, h - box_h - 10

    # sfondo semi-trasparente
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    for i, det in enumerate(targets):
        rank = i + 1
        color = RANK_COLORS[rank]
        maturity = det.get("maturity", 0)
        text = f"#{rank}   G: {det['geo_score']:.2f}   M: {maturity:.0%}   A: {det['area']}"
        ty = y0 + padding + line_h * i + 20
        cv2.putText(image, text, (x0 + padding, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
# pipeline principale

def select_targets(candidates, image_bgr):
    """
    Selezione a due stadi:
      1. Ordina per score geometrico, prendi i primi GEOMETRIC_POOL
      2. Tra questi, filtra per maturità >= soglia
      3. Se rimangono >= MAX_TARGETS, prendi i top MAX_TARGETS
         Altrimenti, completa con i migliori geometrici (fallback)
    """
    # fase 1: ranking geometrico
    candidates.sort(key=lambda x: x["geo_score"], reverse=True)
    pool = candidates[:GEOMETRIC_POOL]

    # fase 2: calcolo maturità e filtro
    for det in pool:
        det["maturity"] = compute_maturity(image_bgr, det["mask"])

    mature = [d for d in pool if d["maturity"] >= MATURITY_THRESHOLD]
    immature = [d for d in pool if d["maturity"] < MATURITY_THRESHOLD]

    selected = mature[:MAX_TARGETS]

    return selected


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

        # fase 1: estrazione metriche geometriche
        candidates = []
        if results.masks is not None:
            for i, mask_tensor in enumerate(results.masks.data):
                if int(results.boxes.cls[i]) != TARGET_CLASS:
                    continue

                m = mask_tensor.cpu().numpy()
                m = cv2.resize(m, (w, h))
                binary = (m > 0.5).astype(np.uint8) * 255

                result = compute_geometric_score(binary, h, w)
                if result is not None:
                    candidates.append(result)

        if not candidates:
            cv2.imwrite(os.path.join(output_dir, f"ranked_{img_name}"), img)
            continue

        # fase 2: selezione con filtro maturità
        top = select_targets(candidates, img)

        # visualizzazione
        output = img.copy()
        for rank, det in enumerate(top, 1):
            color = RANK_COLORS[rank]
            output = overlay_mask(output, det["mask"], color)
            draw_rank_marker(output, det["centroid"], rank, color)
        
        draw_legend(output, top)

        save_path = os.path.join(output_dir, f"ranked_{img_name}")
        cv2.imwrite(save_path, output)

    print(f"Completato. Risultati salvati in: {output_dir}/")


if __name__ == "__main__":
    process_folder()