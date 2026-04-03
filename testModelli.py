import pandas as pd
from ultralytics import YOLO
import torch
import os

# Configurazione modelli (V2 - 800px)
# Aggiungi qui tutti i modelli che desideri testare
model_configs = {
    "Large 11": {
        "path": "yolo11NewData6/weights/best.pt",
        "imgsz": 800,
        "dataset_tag": "NewData"
    }
}

dataset_yaml = "nuovoDatasetTest/data.yaml"
conf_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # Intervallo di confidenza richiesto
results_list = []

# Rilevamento hardware ottimizzato
device = 0 if torch.cuda.is_available() else "cpu"
if device == "cpu" and torch.backends.mps.is_available():
    device = "mps"

print(f"Device operativo: {device}")

for name, cfg in model_configs.items():
    if not os.path.exists(cfg["path"]):
        print(f"\n[!] Errore: Pesi non trovati per {name} in {cfg['path']}")
        continue
    
    # Carichiamo il modello una sola volta per tutte le confidenze (efficienza VRAM)
    model = YOLO(cfg["path"])
    
    for conf_thresh in conf_values:
        print(f"\n>>> Valutazione: {name} | Confidenza: {conf_thresh}")
        
        # Esecuzione validazione
        # verbose=False riduce il rumore nei log durante le iterazioni
        metrics = model.val(
            data=dataset_yaml,
            split='test',   
            task='detect',
            conf=conf_thresh,
            batch=1,
            imgsz=cfg["imgsz"],
            device=device,
            save_json=False,
            verbose=False,
            plots=False # Disattivato per velocizzare l'iterazione
        )
        
        # Estrazione metriche e append alla lista
        results_list.append({
            "Model": name,
            "Dataset": cfg.get("dataset_tag", "N/A"),
            "Box_mAP50": metrics.box.map50,
            "Box_mAP50-95": metrics.box.map,
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr,
            "Conf": conf_thresh
        })

# Elaborazione finale dei dati
if results_list:
    df = pd.DataFrame(results_list)
    df = df.round(4) # Maggiore precisione per analisi scientifica
    
    print("\n" + "="*50)
    print("### REPORT FINALE DI VALIDAZIONE MULTI-CONFIDENZA ###")
    print("="*50)
    
    # Visualizzazione Tabellare
    print("\nVisualizzazione DataFrame:")
    print(df.to_string(index=False))
    
    # Formato CSV per copia-incolla o salvataggio
    print("\nFORMATO CSV (Pronto per .csv):")
    print(df.to_csv(index=False))
    
    # Formato Markdown per il tuo README
    print("\nTABELLA MARKDOWN PER README:")
    print(df.to_markdown(index=False))
    
    # Salvataggio automatico su file
    #df.to_csv("risultati_sensibilita_confidence.csv", index=False)
    #print(f"\n[INFO] Risultati salvati correttamente in 'risultati_sensibilita_confidence.csv'")
else:
    print("Errore: Nessun dato raccolto. Controllare i percorsi dei modelli.")