# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer vision project for training and evaluating YOLO-based models (YOLO11 and YOLO26) for tomato detection and instance segmentation in greenhouse environments. The main application is a ranking algorithm for selecting harvestable tomatoes based on geometric and chromatic features, targeting robotic harvesting systems.

## Setup

No build step. Install dependencies:
```bash
pip install ultralytics opencv-python pandas matplotlib torch
```

No virtual environment is tracked (excluded in `.gitignore`).

## Common Commands

```bash
# Train a model
python train_pomodori.py

# Evaluate model across confidence thresholds (0.2–0.7)
python testModelli.py

# Run reachability ranking on test images
python reachability_ranking.py

# Generate performance charts and Pareto frontier plots
python grafici.py

# Quick single-image inference (edit IMAGE_PATH inside)
python prova.py

# Interactive image browser with predictions (A/D keys to navigate)
python slideshow.py
```

There are no automated tests or linting configurations.

## Architecture

The project has three functional layers:

**1. Training & Evaluation**
- `train_pomodori.py` — Trains YOLO26 Medium on `datasetAggiornato/`; outputs weights to `runs/segment/<run_name>/weights/best.pt`
- `testModelli.py` — Validates models against `nuovoDatasetTest/`; sweeps confidence thresholds and outputs a CSV + Markdown table
- `grafici.py` — Reads `runs/segment/*/results.csv` and generates comparative PNG plots

**2. Ranking Algorithm (core contribution)**
- `reachability_ranking.py` — Two-stage pipeline:
  1. **Geometric score**: 45% area + 35% circularity + 20% centrality (distance from optical center)
  2. **Maturity filter**: HSV-based red/green ratio (threshold 0.4)
  - Outputs color-coded ranked overlays to `output_reachability/`: green (rank 1), orange (rank 2), red-orange (rank 3)

**3. Utilities**
- `prova.py` — Single-image inference for quick sanity checks
- `slideshow.py` — Interactive OpenCV viewer for browsing dataset predictions

## Data Flow

```
datasetAggiornato/ → train_pomodori.py → runs/segment/*/weights/best.pt
                                              ↓
nuovoDatasetTest/  → testModelli.py   → results.csv → grafici.py → PNG charts
                                              ↓
datasetAggiornato/test/images/ → reachability_ranking.py → output_reachability/
```

## Dataset

`datasetAggiornato/data.yaml` defines 4 classes: `6kp_peduncle`, `6kp_petiole`, `MainStem`, `tomato`. Source is Roboflow (workspace: pomodori-gqsky). A secondary dataset `nuovoDatasetTest/` is used exclusively for evaluation.

## Key Notes

- **Best performing model by efficiency**: YOLO11-Large (mAP50-95: 0.5719, 55.8 MB, ~2.5h training) — see README for full comparison tables
- Model weights (`.pt`, `.pth`, `.onnx`) are excluded from git
- `runs/segment/` contains 20+ experimental training runs; previous script versions are archived in `versioniPrecedenti/`
- Training uses cosine annealing LR, dropout 0.3, aggressive augmentation (mosaic, flips, color jitter)

# Tesi Marcassa — Rilevamento e segmentazione del pomodoro con YOLO

## Panoramica progetto

Tesi triennale di informatica (Università degli Studi di Milano). Il progetto confronta modelli YOLO11 e YOLO26 per object detection e instance segmentation di pomodori in serra, nell'ambito di un sistema robotico di raccolta automatizzata.

## Struttura del repo

- Tesi.docx — File della tesi (capitoli, figure, tabelle)

## Modelli addestrati

Sono stati addestrati diverse varianti di YOLO11 e YOLO26 (nano, small, medium, large) sia in modalità detect che segment, con diverse configurazioni:
- Cosine annealing del learning rate
- Batch size variabile
- Risoluzioni spaziali diverse (es. 640, 1024)
- Ottimizzatore MuSGD (YOLO26) vs SGD/AdamW (YOLO11)

## Metriche chiave

Le metriche da estrarre dai risultati sono: mAP50, mAP50-95, Precision, Recall, tempi di inferenza (ms), numero di parametri. Guardare i file `results.csv` dentro ogni cartella `runs/`.

## Algoritmo di ranking e raggiungibilità

È stato sviluppato un algoritmo custom che, data la segmentazione dei pomodori, ne stima la raggiungibilità per il braccio robotico (quali frutti sono accessibili e in che ordine raccoglierli). È stato condotto un sondaggio/valutazione di questo algoritmo.

## Stile di scrittura della tesi

- Italiano accademico ma fluido, NON da chatbot
- Prosa discorsiva con passaggi logici collegati (no elenchi puntati nel corpo del testo)
- Note a piè di pagina per le fonti, non citazioni inline tra parentesi
- Termini tecnici in inglese quando sono standard nel campo (bounding box, feature map, backbone, loss, ecc.)
- Quando si introduce un concetto, spiegarlo brevemente prima di usarlo

## Task comuni

### Estrarre risultati
```bash
# Esempio: leggere le metriche finali da un run
cat runs/detect/train/results.csv | tail -1
```

### Generare grafici
- Usa matplotlib con stile pulito (no griglie pesanti, font leggibili)
- Salva in formato PDF per inserimento nella tesi LaTeX/Word
- Etichette degli assi in italiano

### Scrivere testo per la tesi
- Segui lo stile descritto sopra
- Non ripetere concetti già trattati nei capitoli precedenti
- Fai riferimento ai dati concreti, non a frasi generiche
- Collega sempre il risultato numerico alla sua implicazione pratica per il progetto robotico

## Cosa NON fare

- Non generare testo che sembri scritto da un AI (frasi generiche, struttura a bullet points, tono da blog)
- Non inventare dati: se un file non esiste, segnalalo
- Non usare mai "in conclusione" o "in sintesi" a metà paragrafo
- Non ripetere le stesse informazioni in sezioni diverse