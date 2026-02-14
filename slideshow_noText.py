from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys

folder_path = 'pomodori-1/test/images'  
model_path = 'runs/segment/modello_migliore/weights/best.pt' 

COLOR_MAP = {
    'tomatos':      (0, 0, 255),     # Rosso
    'MainStem':     (255, 0, 0),     # Blu
    'MainStems':    (255, 0, 0),     # Blu (caso plurale)
    '6kp_petioles': (0, 255, 255),   # Giallo
    '6kp_peduncle': (255, 0, 255),   # Viola/Magenta
    'default':      (128, 128, 128)  # Grigio per cose non previste
}
ALPHA = 0.75  # Trasparenza 

if not os.path.exists(model_path):
    print(f"ERRORE: Non trovo il modello in {model_path}")
    sys.exit()

model = YOLO(model_path)
names = model.names  

images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
images.sort()
current_index = 0

def ridimensiona_smart(img):
    return img if img.shape[:2] == (432, 432) else cv2.resize(img, (432, 432))

def disegna_maschere_personalizzate(image, results):
    """
    Disegna le maschere manualmente senza etichette di testo.
    """
    # Copia dell'immagine per creare l'overlay trasparente
    overlay = image.copy()
    
    # Se non trova nulla, restituisci l'immagine pulita
    if results[0].masks is None:
        return image

    # Dati rilevati
    masks_data = results[0].masks.xy     # Le coordinate dei poligoni
    classes_ids = results[0].boxes.cls   # Gli ID delle classi trovate

    for polygon, class_id in zip(masks_data, classes_ids):
        # 1. Trova il nome della classe
        class_name = names[int(class_id)]
        
        # 2. Scegli il colore
        color = COLOR_MAP['default']
        # Cerca se una delle chiavi della nostra mappa è contenuta nel nome della classe
        for key in COLOR_MAP:
            if key in class_name:
                color = COLOR_MAP[key]
                break
        
        # 3. Converti il poligono in formato leggibile per OpenCV (int32)
        polygon = np.array(polygon, dtype=np.int32)
        
        # 4. Disegna il poligono PIENO sull'overlay
        cv2.fillPoly(overlay, [polygon], color)
        
        # (Opzionale) Disegna anche il contorno più marcato
        cv2.polylines(overlay, [polygon], True, color, 2)

    # 5. Fondi l'immagine originale con l'overlay (Trasparenza)
    return cv2.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0)

print("--- VISUALIZZATORE CUSTOM ---")
print(" [ D ] -> Avanti")
print(" [ A ] -> Indietro")
print(" [ Q ] -> Esci")

while True:
    img_name = images[current_index]
    img_path = os.path.join(folder_path, img_name)
    
    frame = cv2.imread(img_path)
    if frame is None:
        current_index = (current_index + 1) % len(images)
        continue

    # 1. Ridimensiona
    frame_resized = ridimensiona_smart(frame)

    # 2. Predizione
    results = model.predict(frame_resized, conf=0.25, verbose=False)

    # 3. Disegno Manuale (La magia accade qui)
    final_frame = disegna_maschere_personalizzate(frame_resized, results)

    # Info di navigazione (piccolo in alto)
    cv2.putText(final_frame, f"{current_index+1}/{len(images)}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Analisi Pianta", final_frame)

    key = cv2.waitKey(0)
    if key == ord('q'): break
    elif key == ord('d') or key == 83: current_index = (current_index + 1) % len(images)
    elif key == ord('a') or key == 81: current_index = (current_index - 1) % len(images)

cv2.destroyAllWindows()