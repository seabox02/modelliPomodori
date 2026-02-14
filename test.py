from ultralytics import YOLO
import cv2
import sys

# Percorso immagine (da terminale o default)
image_path = sys.argv[1] if len(sys.argv) > 1 else 'home/aislab/Desktop/modelliDeepLearning/pomodori-1/test/images/2621-60-120_color_jpg.rf.e5b9a6dc95ffc7d8060fb90360e9b080.jpg'


# Carica modello
model = YOLO('runs/segment/modello_pomodori/weights/best.pt') 

# 1. FAI L'INFERENZA (Senza show=True)
results = model.predict(image_path, conf=0.4)

# 2. DISEGNA I RISULTATI MANUALMENTE
# Prendi il primo risultato (visto che gli passiamo una sola foto)
r = results[0]
# .plot() crea l'immagine con le maschere disegnate sopra
annotated_frame = r.plot()

# 3. MOSTRA LA FINESTRA
cv2.imshow("Risultato YOLO", annotated_frame)

print("Premi un tasto qualsiasi sulla finestra dell'immagine per chiudere...")

# Aspetta finché non premi un tasto
cv2.waitKey(0)
cv2.destroyAllWindows()