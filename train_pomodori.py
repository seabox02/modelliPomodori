from ultralytics import YOLO

# 1. Carica il modello
model = YOLO('runs/segment/modello_pomodori3/weights/best.pt')  

# 2. Avvia l'addestramento
# device=0 forza l'uso della tua GTX 1060
model.train(
    data='/home/aislab/Desktop/modelliDeepLearning/pomodori-1/data.yaml',  # Punta al file scaricato da Roboflow
    epochs=150,                 # Proviamo con 50 giri per iniziare
    imgsz=640,                 # Dimensione immagini
    batch=4,                   # Se la GPU va in "Out of Memory", abbassa a 4
    patience=30,
    augment=True,
    dropout=0.1,
    device=0,
    name='modello_pomodori4'    # Nome del salvataggio
)