"""
Script di confronto avanzato YOLO vs Ground Truth (GT).
Indica se i box/maschere sono presenti nelle annotazioni di test.

Colori:
  - VERDE: Predizione corretta (presente nei test)
  - ROSSO: Predizione extra (Falso Positivo - non presente nei test)
  - BLU: Oggetto mancato (Falso Negativo - presente nei test ma ignorato dal modello)

Comandi:
  - A / D : Immagine precedente / successiva
  - M : Mostra/Nascondi Maschere (segmentazione)
  - B : Mostra/Nascondi Bounding Box
  - L : Mostra/Nascondi Etichette di stato (OK / NOT IN TEST)
  - G : Mostra/Nascondi oggetti mancati (Blu)
  - Q : Esci
"""

import cv2
import numpy as np
import glob
import os
from ultralytics import YOLO

# === CONFIGURAZIONE ===
MODEL_PATH = "runs/segment26/large/modello26_large_150-32_DA/weights/best.pt"
TEST_IMAGES = "pomodoriData/test/images"
TEST_LABELS = "pomodoriData/test/labels"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.40  # Soglia per match tra predizione e test
# =======================

def get_iou(boxA, boxB):
    """Calcola l'Intersection over Union (IoU) tra due box [x1, y1, x2, y2]."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = float(boxAArea + boxBArea - interArea)
    return interArea / union if union > 0 else 0

def load_gt_boxes(img_path, img_w, img_h):
    """Carica le annotazioni reali (box o poligoni)."""
    basename = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(TEST_LABELS, basename + ".txt")
    gt_data = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5: continue
                coords = parts[1:]
                if len(coords) > 4: # Poligono (x1 y1 x2 y2 ...)
                    xs, ys = coords[0::2], coords[1::2]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                else: # Box YOLO (xc yc w h)
                    xc, yc, w, h = coords
                    x1, y1, x2, y2 = xc - w/2, yc - h/2, xc + w/2, yc + h/2
                gt_data.append({'box': [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]})
    return gt_data

# Inizializzazione
print(f"Caricamento modello: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

image_paths = sorted(glob.glob(os.path.join(TEST_IMAGES, "*.[jJ][pP][gG]")) +
                     glob.glob(os.path.join(TEST_IMAGES, "*.[pP][nN][gG]")))

if not image_paths:
    print(f"Errore: Nessuna immagine in {TEST_IMAGES}")
    exit(1)

print("Eseguo analisi su tutte le immagini...")
results_cache = []
for i, path in enumerate(image_paths):
    res = model(path, conf=CONF_THRESHOLD, verbose=False)[0]
    h, w = res.orig_shape
    gt = load_gt_boxes(path, w, h)
    results_cache.append({'res': res, 'gt': gt})

# Stato visualizzazione
idx = 0
show_masks = True
show_boxes = True
show_labels = True
show_missing = True

def draw_frame(index):
    data = results_cache[index]
    img = cv2.imread(image_paths[index])
    h, w = img.shape[:2]
    overlay = img.copy()
    
    res = data['res']
    gt_list = [item['box'] for item in data['gt']]
    gt_matched = [False] * len(gt_list)
    
    # 1. Calcola i match predizione -> GT
    pred_matches = []
    if res.boxes is not None:
        for p_box in res.boxes.xyxy.cpu().numpy():
            best_iou, match_idx = 0, -1
            for g_i, g_box in enumerate(gt_list):
                if gt_matched[g_i]: continue
                iou = get_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou, match_idx = iou, g_i
            
            is_matched = best_iou >= IOU_THRESHOLD
            if is_matched: gt_matched[match_idx] = True
            pred_matches.append(is_matched)

    # 2. Disegna Maschere
    if show_masks and res.masks is not None:
        for i, mask in enumerate(res.masks.data):
            color = (0, 255, 0) if pred_matches[i] else (0, 0, 255)
            m_np = cv2.resize(mask.cpu().numpy(), (w, h)) > 0.5
            colored_mask = np.zeros_like(img)
            colored_mask[m_np] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.35, 0)

    # 3. Disegna Box e Label
    if res.boxes is not None:
        for i, p_box in enumerate(res.boxes.xyxy.cpu().numpy()):
            color = (0, 255, 0) if pred_matches[i] else (0, 0, 255)
            x1, y1, x2, y2 = map(int, p_box)
            
            if show_boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            if show_labels:
                lbl = f"{'OK' if pred_matches[i] else 'NOT IN TEST'} {res.boxes.conf[i]:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(overlay, (x1, y1-th-4), (x1+tw, y1), color, -1)
                cv2.putText(overlay, lbl, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # 4. Disegna mancanti (Blu)
    if show_missing:
        for i, g_box in enumerate(gt_list):
            if not gt_matched[i]:
                gx1, gy1, gx2, gy2 = map(int, g_box)
                cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (255, 0, 0), 1)
                if show_labels:
                    cv2.putText(overlay, "MISSING", (gx1, gy1+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Barra UI
    ui = np.zeros((65, overlay.shape[1], 3), dtype=np.uint8)
    fname = os.path.basename(image_paths[index])
    cv2.putText(ui, f"[{index+1}/{len(image_paths)}] {fname}", (10, 25), 0, 0.5, (255,255,255), 1)
    status = f"M:Mask({show_masks}) B:Box({show_boxes}) L:Label({show_labels}) G:Missing({show_missing})"
    cv2.putText(ui, status, (10, 50), 0, 0.4, (180,180,180), 1)
    return np.vstack([ui, overlay])

# Loop principale
cv2.namedWindow("Confronto YOLO vs GT", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Confronto YOLO vs GT", 1280, 850)

while True:
    cv2.imshow("Confronto YOLO vs GT", draw_frame(idx))
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q') or key == 27: break
    elif key == ord('d'): idx = (idx + 1) % len(image_paths)
    elif key == ord('a'): idx = (idx - 1) % len(image_paths)
    elif key == ord('m'): show_masks = not show_masks
    elif key == ord('b'): show_boxes = not show_boxes
    elif key == ord('l'): show_labels = not show_labels
    elif key == ord('g'): show_missing = not show_missing

cv2.destroyAllWindows()
