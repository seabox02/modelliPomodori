from ultralytics import YOLO

# 1. Carica il modello
model = YOLO('yolo11n-seg.pt')  

# 2. Avvia l'addestramento
# device=0 forza l'uso della tua GTX 1060
model.train(
    data='/home/aislab/Desktop/modelliDeepLearning/pomodori-1/data.yaml',  # Punta al file scaricato da Roboflow
    epochs=50,                 # Proviamo con 50 giri per iniziare
    imgsz=640,                 # Dimensione immagini
    batch=8,                   # Se la GPU va in "Out of Memory", abbassa a 4
    device=0,
    name='modello_pomodori'    # Nome del salvataggio
)