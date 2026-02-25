from ultralytics import YOLO
import cv2
import os

folder_path = 'datasetTest/test' 
model = YOLO('runs/segment/modello_extralarge_150/weights/best.pt') # per visualizzare un altro modello modificare la directory con quella del modello voluto


images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]
current_index = 0

def ridimensiona_smart(img):
   return img if img.shape[:2] == (432, 432) else cv2.resize(img, (432, 432))

print("COMANDI:")
print(" [ D ] -> Prossima immagine")
print(" [ A ] -> Immagine precedente")
print(" [ Q ] -> Esci")

while True:
    img_name = images[current_index]
    
    img_path = os.path.join(folder_path, img_name)
    
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f"Impossibile leggere: {img_name}")
        current_index = (current_index + 1) % len(images)
        continue

    frame_resized = ridimensiona_smart(frame)

    results = model.predict(frame_resized, conf=0.40, verbose=False)

    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f"Img: {current_index+1}/{len(images)} - {img_name}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Slideshow Pomodori (Auto-Size)", annotated_frame)

    key = cv2.waitKey(0) 

    if key == ord('q'): 
        break
    elif key == ord('d'): 
        current_index = (current_index + 1) % len(images)
    elif key == ord('a'): 
        current_index = (current_index - 1) % len(images)

cv2.destroyAllWindows()