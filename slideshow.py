from ultralytics import YOLO
import cv2
import os

folder_path = 'pomodori-1/valid/images' 
model = YOLO('runs/segment/modello_pomodori/weights/best.pt')


images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
current_index = 0

print("COMANDI:")
print(" [ D ] -> Prossima immagine")
print(" [ A ] -> Immagine precedente")
print(" [ Q ] -> Esci")

while True:
    img_name = images[current_index]
    img_path = os.path.join(folder_path, img_name)
    
    results = model.predict(img_path, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f"Img: {current_index+1}/{len(images)} - {img_name}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Navigatore Pomodori", annotated_frame)

    key = cv2.waitKey(0) 

    if key == ord('q'): 
        break
    elif key == ord('d'): 
        current_index = (current_index + 1) % len(images)
    elif key == ord('a'): 
        current_index = (current_index - 1) % len(images)

cv2.destroyAllWindows()