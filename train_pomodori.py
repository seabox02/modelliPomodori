from ultralytics import YOLO

model = YOLO('yolo26m-seg.pt')

model.train(
    data='datasetAggiornato/data.yaml',
    name='medium26_200_32',
    epochs=200,
    imgsz=640,
    batch=32,
    patience=30,
    save=True,
    plots=True,
    cache=True, #carica le immagini in ram utile visto che non ne abbiamo tantissime

#Ottimizzazione e stabilità
    dropout=0.3,
    optimizer='auto',
    lr0=0.001,
    lrf=0.01,
    cos_lr = True,

#Agumentation, posizionale
    fliplr=0.5,
    flipud=0.5,
    degrees=90.0,
    scale=0.5,
    translate=0.1,

#Agumentation, colore e luce
    hsv_s=0.7,
    hsv_h = 0.015,
    hsv_v=0.4,
#Agumentation, complessità
    mosaic=1.0,
#Rimosso mixup perche produceva nan in validation
    copy_paste=0.4,
    erasing=0.4,

    close_mosaic = 10 #spegne mosaic ultime 10 epoche
)