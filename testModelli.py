import pandas as pd
from ultralytics import YOLO
import torch
import os

# Definiamo solo i modelli a 800px (Generazione V2) disponibili
model_configs = {
    "Small V2 (800px)": {
        "path": "runs/segment/s26Finale/weights/best.pt",
        "imgsz": 800
    },
    "Medium (800px)": {
        "path": "runs/segment/m26Finale/weights/best.pt",
        "imgsz": 800
    },
    "Large V2 (800px)": {
        "path": "runs/segment/Nuovol26/weights/best.pt",
        "imgsz": 800
    },
    "ExtraLarge (800px)": {
        "path": "runs/segment/NuovoXL/weights/best.pt",
        "imgsz": 800
    }
}

dataset_yaml = "datasetAggiornato/data.yaml"
results_list = []

# Rilevamento hardware
device = 0 if torch.cuda.is_available() else "cpu"
if device == "cpu" and torch.backends.mps.is_available():
    device = "mps"

print(f"Utilizzo device: {device}")

for name, cfg in model_configs.items():
    if not os.path.exists(cfg["path"]):
        print(f"\n[!] Attenzione: Pesi non trovati per {name} in {cfg['path']}")
        continue
        
    folder_name = f"val_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
    print(f"\n>>> Valutazione Test Set: {name}")
    
    model = YOLO(cfg["path"])
    
    # Validazione sul Test Set
    metrics = model.val(
        data=dataset_yaml,
        split='test',   
        conf=0.2,
        batch=1,
        imgsz=cfg["imgsz"],
        device=device,
        save_json=True,
        verbose=False,
        name=folder_name,
        exist_ok=True
    )
    
    results_list.append({
        "Modello": name,
        "Box mAP50": metrics.box.map50,
        "Box mAP50-95": metrics.box.map,
        "Mask mAP50": metrics.seg.map50,
        "Mask mAP50-95": metrics.seg.map,
        "Precision (M)": metrics.seg.mp,
        "Recall (M)": metrics.seg.mr,
    })

# Output e salvataggio
if results_list:
    df = pd.DataFrame(results_list)
    df = df.round(3)
    
    print("\n### RISULTATI TEST SET (GENERAZIONE V2 - 800px) ###")
    print(df.to_string(index=False))
    
    print("\nTabella Markdown per il README:")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))
    
    df.to_csv("risultati_test_finali.csv", index=False)
else:
    print("Nessun modello valutato. Verifica i percorsi dei file .pt")
