## Modelli Object Detection e Mask Segmentation su Dataset Pomodori (YOLO26)

Questo progetto documenta lo sviluppo e l'addestramento di modelli di visione artificiale ottimizzati per il riconoscimento e la segmentazione di istanze di pomodori. Sono state impiegate architetture **YOLO** (YOLO11 e YOLO26) di diverse scale per bilanciare latenza e precisione in contesti robotici.

### Evoluzione della Sperimentazione
Le fasi di addestramento sono state divise in due generazioni principali:
1. **Generazione V1 (640px)**: Baseline standard di YOLO.
2. **Generazione V2 (800px)**: Ottimizzazione della risoluzione di input, fondamentale per risolvere dettagli in scene con occlusioni, unita all'uso di **Retina Masks** per la massima precisione dei contorni.

Entrambe le generazioni beneficiano della **Cosine Annealing Learning Rate (cos_lr)** per una convergenza stabile.

### Risultati Comparativi (Dataset Aggiornato)
I modelli sono stati valutati su 200 epoche. Il confronto evidenzia come l'incremento della risoluzione (V2) porti a un salto di qualità trasversale a tutte le architetture.

| Modello | Imgsz | mAP50 (B) | mAP50-95 (B) | mAP50 (M) | mAP50-95 (M) | Precision (M) | Recall (M) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small V1** | 640 | 0.703 | 0.414 | 0.627 | 0.299 | 0.730 | 0.572 |
| **Small V2** | **800** | 0.736 | 0.451 | 0.663 | 0.346 | 0.728 | 0.636 |
| **Medium** | 800 | 0.767 | 0.477 | 0.700 | 0.367 | 0.755 | 0.645 |
| **Large V1** | 640 | 0.754 | 0.464 | 0.685 | 0.344 | 0.717 | 0.631 |
| **Large V2 (Best)** | **800** | **0.778** | **0.491** | **0.712** | **0.373** | **0.762** | **0.670** |
| **ExtraLarge** | 800 | 0.775 | 0.485 | 0.711 | 0.370 | 0.737 | 0.667 |

*(B) = Bounding Box, (M) = Segmentation Mask. Valutazioni eseguite con confidenza 0.20.*

#### Evoluzione delle Metriche (mAP@.50:.95)
![Confronto Bounding Box](grafici/confronto_bbox_95.png)
*Figura 1: Evoluzione della precisione media (mAP50-95) per le Bounding Box.*

![Confronto Segmentation Mask](grafici/confronto_mask_95.png)
*Figura 2: Evoluzione della precisione media (mAP50-95) per le Maschere di Segmentazione.*

### Analisi Tecnica
1. **L'impatto della Risoluzione (V1 vs V2)**: Il passaggio da x640 a 800 pixel è il fattore determinante. Il modello **Small V2** (800px) riesce a superare le prestazioni del **Large V1** (640px) pur avendo un numero di parametri significativamente inferiore, a dimostrazione che la densità di pixel è critica per la segmentazione in questo dominio.
2. **Equilibrio Architetturale**: Il modello **Large V2 (800px)** rappresenta l'ottimo di Pareto: offre prestazioni superiori alla versione ExtraLarge (XL) con una complessità ridotta, confermando che l'architettura Large ha già la capacità necessaria per il dataset attuale.

### Analisi della Convergenza e Generalizzazione
L'analisi delle curve evidenzia una robusta capacità di generalizzazione:
- **Assenza di Overfitting**: La stabilità della loss di validazione conferma che i modelli non hanno "memorizzato" il training set.
- **Ottimizzazione**: L'impiego del Cosine Annealing ha stabilizzato la precisione finale al valore massimo (0.373 mAP50-95 per le maschere nel modello Large V2).

![Analisi Overfitting](grafici/analisi_overfitting_seg.png)
*Figura 3: Analisi della convergenza tramite Validation Segmentation Loss.*

### Validazione Finale su Test Set (Unbiased Evaluation) con una confidenza di 0.4:

I modelli sono stati valutati su un test set indipendente con confidenza fissata a 0.4, per una stima unbiased delle performance reale.

| Modello                    |   Box mAP50 |   Box mAP50-95 |   Mask mAP50 |   Mask mAP50-95 |   Precision (M) |   Recall (M) |
|:---------------------------|------------:|---------------:|-------------:|----------------:|----------------:|-------------:|
| Small NewData (800px)      |       0.74  |          0.493 |        0.682 |           0.379 |           0.786 |        0.567 |
| Medium NewData (800px)     |       0.77  |          0.521 |        0.722 |           0.397 |           0.787 |        0.626 |
| Large NewData (800px)      |       0.778 |          0.532 |        0.724 |           0.407 |           0.78  |        0.628 |
| **ExtraLarge NewData (800px)** |       **0.796** |          **0.541** |        **0.743** |           0.408 |           0.778 |        0.674 |
| Small OldData (800px)      |       0.746 |          0.499 |        0.697 |           0.386 |           0.787 |        0.587 |
| Medium OldData (800px)     |       0.751 |          0.515 |        0.705 |           0.398 |           0.768 |        0.595 |
| Large OldData (800px)      |       0.775 |          0.533 |        0.719 |           0.399 |           0.787 |        0.621 |
| ExtraLarge OldData (800px) |       0.779 |          0.54  |        0.713 |           0.403 |           0.761 |        0.648 |
| ExtraLarge 11 (800px)      |       0.803 |          0.576 |        0.738 |           0.411 |            0.76 |        0.688 |
| Medium 11  (640px)   | 0.7957 |         0.5615 |       0.7167 |          0.4014 |      0.7226 |   0.6567 |
| **Large 11 (640px)** |      **0.8068** |         **0.5719** |       **0.732**  |          **0.4123** |      **0.75**   |   **0.6813** |
| ExtraLarge 11 (640px)|      0.8028 |         0.5756 |       0.7405 |          0.429  |      0.7594 |   0.6896 |

In [risultati](grafici/risultati.csv) troviamo tutti i valori ottenuti su vari livelli di confidence (da 0.2 a 0.7).

Nonostante YOLO26 sia l'architettura più recente e teoricamente più performante, il modello **Large di YOLO11** ottiene performance molto simili, e migliori su alcune metriche, rispetto all'ExtraLarge di YOLO26.

![mAP50-95](grafici/compare1.png) ![precision](grafici/compare2.png) ![recall](grafici/compare3.png)

I grafici evidenziano tre aspetti principali:
- **mAP50-95**: i due modelli sono sostanzialmente equivalenti su tutti i valori di confidence; 
- **Precision**: YOLO26-XL raggiunge una precisione più elevata ad alta confidence, vicina al 90%, mentre YOLO11-L è più contenuto, ma non c'è una differenza così netta;
- **Recall**: YOLO11-L mantiene una recall significativamente più stabile all'aumentare della confidence, mentre YOLO26-XL diminuisce drasticamente.

La stabilità delle metriche al variare della confidence è un punto di forza per applicazioni reali: consente di adattare la soglia in base al caso d'uso — ad esempio privilegiare la recall per raccogliere il maggior numero di pomodori, o la precision per evitare falsi positivi in raccolta automatizzata.

### Analisi dell'Efficienza e Costo Computazionale mAP50-95 (Mask)
| Modello | Peso (MB) | Tempo Training | mAP50-95 (Mask) | Efficienza |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO26-S NewData** | 23.4 MB | ~1h 50m | 0.379 | Alta Efficienza |
| **YOLO11-M OldData** | 45.2 MB | ~2h 10m | 0.401 | Ottimo bilanciamento |
| **YOLO11-L OldData** | 55.8 MB | ~2h 40m | 0.412 |  **Best Performance** |
| **YOLO26-M NewData** | 54.5 MB | ~2h 50m | 0.397 | Buona efficienza |
| **YOLO26-L NewData** | 63.5 MB | ~3h 20m | 0.407 | Superato da YOLO11-L |
| **YOLO11-XL OldData** | 124.8 MB | ~4h 30m | **0.429** | Inefficiente |
| **YOLO26-XL NewData** | 141.8 MB | ~5h 30m | 0.408 | Inefficiente |

![pareto_mAp50-95](grafici/pareto_mAP50-95.png)

*Figura 4: Analisi di efficienza. La dimensione della bolla rappresenta il peso del modello in MB.*

### Analisi dell'Efficienza e Costo Computazionale mAP50 (Mask)
| Modello | Peso (MB) | Tempo Training | mAP50 (Mask) | Efficienza |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO26-S NewData** | 23.4 MB | 1h 50m | 0.682 | Alta Efficienza |
| **YOLO11-M OldData** | 45.2 MB | 2h 10m | 0.717 | Buona Efficienza |
| **YOLO11-L OldData** | 55.8 MB | 2h 40m | 0.732 | **Best Performance** |
| **YOLO26-M NewData** | 54.5 MB | 2h 50m | 0.722 | Ottimo bilanciamento |
| **YOLO26-L NewData** | 63.5 MB | 3h 20m | 0.724 |  Superato da YOLO11-L |
| **YOLO11-XL OldData** | 124.8 MB | 4h 30m | 0.741 | Inefficiente |
| **YOLO26-XL NewData** | 141.8 MB | 5h 30m | **0.743** | **Inefficiente** |

![pareto_mAP50](grafici/pareto_mAP50.png)

In entrambe le metriche emerge lo stesso risultato: i modelli ExtraLarge sono i migliori a livello di precisione assoluta, YOLO11-XL per mAP50-95 e YOLO26-XL per mAP50, ma entrambi risultano inefficienti in termini di peso e tempo di training. Il miglior compromesso è **YOLO11-L**, che con un tempo di addestramento contenuto (~2h 40m) e un peso di 55.8 MB raggiunge performance elevate su entrambe le metriche.

### Evoluzione dell'Algoritmo di Selezione Target (reachability_ranking.py)

Lo sviluppo del modulo di selezione dei target ha seguito un percorso incrementale, passando da un'euristica dimensionale a un ranking geometrico complesso.

#### Fase 1: Baseline di Reachability (Area-Based)
Inizialmente, il sistema utilizzava l'**Area della Maschera** come unico criterio di selezione:
- **Logica**: L'area veniva usata come *proxy* della distanza e dell'assenza di occlusioni.
- **Ranking**: I pomodori venivano ordinati per area decrescente, selezionando i Top 3.
- **Limiti**: Un pomodoro grande ma fortemente occluso da foglie (forma a mezzaluna) veniva comunque prioritizzato, rischiando di fornire coordinate errate al braccio robotico per il calcolo del centroide.

#### Fase 2: Sistema Attuale di Graspability Ranking
Per superare i limiti della Fase 1, è stato implementato un sistema di **Graspability Ranking** basato su una funzione di costo multi-obiettivo. Questo modulo filtra i target rilevati da YOLO per identificare i candidati più idonei alla raccolta.

##### Logica di Funzionamento: Graspability Score (G)
Ogni istanza viene valutata con un punteggio $G \in [0, 1]$, calcolato come:
$$G = 0.4 \cdot Area_{norm} + 0.4 \cdot Circolarità + 0.2 \cdot Centralità$$

1.  **Area Normalizzata (40%)**: Proxy della distanza e della visibilità.
2.  **Circolarità (40%)**: Calcolata come $C = \frac{4\pi \cdot Area}{Perimetro^2}$. È il parametro critico per mitigare le **occlusioni**: un pomodoro coperto da foglie presenta una forma irregolare (bassa circolarità), venendo declassato nel ranking per evitare tentativi di presa su geometrie parziali.
3.  **Centralità (20%)**: Valuta la distanza del centroide, che è il baricentro geometricodella maschera, dal centro dell'ottica per minimizzare le distorsioni radiali. È una stima iniziale per il target.

#### Validazione Visiva e Prioritizzazione
Il sistema classifica i **Top 3** target:
-   **Target Ottimale (Verde)**: Score > 0.5. Frutto ben visibile, sferico e in posizione favorevole.
-   **Target Sub-ottimale (Arancio)**: Score < 0.5. Presenza di forti occlusioni o posizione periferica.

#### Fase 3: Occlusion Handling e Priorità di Profondità (Versione Corrente)
Per risolvere i casi critici in cui target sferici e centrali venivano prioritizzati nonostante si trovassero in secondo piano (nascosti da altri frutti), l'algoritmo è stato ulteriormente evoluto con una logica di **Analisi delle Intersezioni**:

- **Rilevamento Sovrapposizioni (IoU)**: Il sistema calcola l'indice *Intersection over Union* tra tutti i bounding box rilevati. Se due pomodori presentano una sovrapposizione significativa (IoU > 0.2), viene attivata la logica di competizione.
- **Occlusion Penalty**: In caso di sovrapposizione, il pomodoro con l'area minore riceve una **penalità del 30%** sullo score finale. Questo assume che, in una proiezione 2D, l'oggetto parzialmente coperto o più lontano appaia più piccolo rispetto a quello in primo piano che lo occulta.
- **Filtro di Magnitudo**: La soglia minima di area è stata elevata a **1500 pixel** per eliminare il "rumore visivo" causato da frutti molto lontani o troppo piccoli per una scansione di qualità.
- **Pesi Bilanciati**: L'area (proxy della vicinanza) assume il peso dominante (50%), seguita dalla circolarità (30%) e dalla centralità (20%).

**Risultato**: Il sistema garantisce che il **Rank 1** sia sempre assegnato al frutto più grande, visibile e "libero" da ingombri frontali, ottimizzando drasticamente la qualità dei dati per la successiva fase di scansione 3D.

### Glossario delle Metriche
- **mAP50-95**: Metrica rigorosa che valuta la precisione millimetrica dei contorni.
- **Precision/Recall**: Capacità di evitare falsi positivi e individuare tutti gli oggetti.

### Dataset e Strumenti
Dataset gestito via [Roboflow](https://roboflow.com/). Log completi in `runs/segment/`.

### Prospettive di Sviluppo Futuro
L'architettura attuale è progettata per essere modulare e supportare due principali direzioni evolutive, a seconda dei requisiti operativi del sistema robotico:

1.  **Direzione Harvesting (Raccolta Robotizzata)**:
    *   **Grasp Point Estimation**: Sfruttare le classi `6kp_peduncle` (peduncolo) e `MainStem` (fusto) già identificate per definire i punti di taglio ottimali, minimizzando lo stress meccanico sulla pianta (approccio ispirato a *StarBL-YOLO*).
    *   **Pose Estimation 6-DoF**: Utilizzo dei dati RGB-D per definire l'orientamento spaziale del frutto e pianificare traiettorie di approccio del gripper che evitino collisioni con la struttura della serra.

2.  **Direzione Phenotyping (Ispezione e Scansione)**:
    *   **Ricostruzione 3D e Analisi del Volume**: Integrazione di modelli di **Amodal Instance Segmentation** per completare la geometria dei frutti parzialmente occlusi, permettendo una stima accurata del volume, della biomassa e della resa produttiva.
    *   **Active Vision (Next Best View)**: Sviluppo di algoritmi per muovere autonomamente il braccio robotico (eye-in-hand) attorno al target, identificando i punti di vista che massimizzano la risoluzione della scansione e l'accuratezza del *Maturity Grading* (grado di maturazione).
