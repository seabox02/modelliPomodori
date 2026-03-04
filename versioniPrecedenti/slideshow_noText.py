from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys

folder_path = 'datasetTest/test' 
model_path = 'runs/segment/modello_medium_ibrido/weights/best.pt' # per visualizzare un altro modello modificare la directory con quella del modello voluto

COLOR_MAP = {
    'tomato':      (0, 0, 255),     # Rosso
    'MainStem':     (255, 0, 0),     # Blu
    'MainStems':    (255, 0, 0),     # Blu 
    '6kp_petioles': (0, 255, 255),   # Giallo
    '6kp_peduncle': (255, 0, 255),   # Viola/Magenta
    'default':      (128, 128, 128)  # Grigio per cose non previste
}
ALPHA = 0.50  # trasparenza 

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
    overlay = image.copy()
    
    if results[0].masks is None:
        return image

    masks_data = results[0].masks.xy     
    classes_ids = results[0].boxes.cls   

    for polygon, class_id in zip(masks_data, classes_ids):
        class_name = names[int(class_id)]
        
        color = COLOR_MAP['default']

        for key in COLOR_MAP:
            if key in class_name:
                color = COLOR_MAP[key]
                break
        
        polygon = np.array(polygon, dtype=np.int32)
        
        cv2.fillPoly(overlay, [polygon], color)
        
        cv2.polylines(overlay, [polygon], True, color, 2)

    return cv2.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0)

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

    frame_resized = ridimensiona_smart(frame)

    results = model.predict(frame_resized, conf=0.25, verbose=False)

    final_frame = disegna_maschere_personalizzate(frame_resized, results)

    cv2.putText(final_frame, f"{current_index+1}/{len(images)}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Analisi Pianta", final_frame)

    key = cv2.waitKey(0)
    if key == ord('q'): break
    elif key == ord('d') or key == 83: current_index = (current_index + 1) % len(images)
    elif key == ord('a') or key == 81: current_index = (current_index - 1) % len(images)

cv2.destroyAllWindows()