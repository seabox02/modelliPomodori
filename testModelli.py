import pandas as pd
from ultralytics import YOLO
import torch

model_paths = {
    "YOLO26l_B16": "runs/segment26/large/modello26_large_150/weights/best.pt",
    "YOLO26l_B32": "runs/segment26/large/modello26_large_150_32/weights/best.pt",
    "YOLO26l_B64": "runs/segment26/large/modello26_large_150_64/weights/best.pt"
}

dataset_yaml = "datasetPomodori/data.yaml"
results_list = []

for name, path in model_paths.items():
    print(f"\n--- Valutazione Modello: {name} ---")
    
    model = YOLO(path)
    
    metrics = model.val(
        data=dataset_yaml,
        split='test',       
        batch=1,            # Batch size 1 per massima precisione per immagine
        imgsz=448,          # Coerente con il training
        device=0,           # Utilizza la tua GPU locale
        save_json=True,
        verbose=False
    )
    
    results_list.append({
        "Modello": name,
        "mAP50": metrics.seg.map50,
        "mAP50-95": metrics.seg.map,
        "Precision": metrics.seg.mp,
        "Recall": metrics.seg.mr,
    })

# Creazione Tabella Comparativa
df = pd.DataFrame(results_list)
print("\n### Risultati Comparativi su Test Set ###")
print(df.to_string(index=False))


df = df.round(3)
markdown_table = df.to_markdown(index=False)
print(markdown_table)

# Salvataggio in CSV per il tuo paper/report
# df.to_csv("test_comparison_results.csv", index=False)