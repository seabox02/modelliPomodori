from ultralytics import YOLO

model = YOLO('yolo11m-seg.pt') 

model.train(
    data='datasetPomodori/data.yaml',
    epochs=50,               
    imgsz=640,
    batch=4,                   
    patience=20,
    augment=True,
    device=0,
    name='nuovoModello'
)