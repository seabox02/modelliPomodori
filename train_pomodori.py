from ultralytics import YOLO

model = YOLO('yolo26m-seg.pt') 

model.train(
    data='datasetPomodori/data.yaml',
    epochs=50,               
    imgsz=640,
    batch=2,                   
    patience=20,
    augment=True,
    device=0,
    name='nuovoModello'
)