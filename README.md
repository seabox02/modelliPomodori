## Modelli object detection e mask segmentation su dataset pomodori (YOLO26)

Nel seguente progetto sono stati sviluppati nuovi modelli di riconoscimento e segmentazione su dataset di pomodori utilizzando **YOLO26** con architetture di diverse dimensioni (small, large, extraLarge) addestrate su **200 epoche** e **batch size 32**. 

I modelli precedenti sviluppati con YOLO26 si possono trovare [qui](versioniPrecedenti/runs/segment26).

 È stata aggiunta la **cosine annealing learning rate** (cos_lr) che aiuta a migliorare la convergenza del modello durante l'allenamento.

### Evoluzione delle metriche durante l'addestramento

Nel seguente grafico vediamo l'evoluzione delle metriche di **Bounding Box** durante i 200 epoch di addestramento. Possiamo notare che il modello **Large** mantiene prestazioni stabili e coerenti, mentre il modello **Small** mostra una convergenza leggermente inferiore.

![Grafico Evoluzione Bounding Box](runs/segment/evolution_chart_bbox_200ep.png)

Nel seguente grafico invece vediamo l'evoluzione delle metriche di **Mask Segmentation**, che mostrano come il modello Large sia significativamente più bravo nel predire i contorni precisi della maschera rispetto agli altri modelli.

![Grafico Evoluzione Mask Segmentation](runs/segment/evolution_chart_mask_200ep.png)

### Risultati finali dei modelli

Nella tabella seguente visualizziamo le metriche finali riguardo le **maschere di segmentazione** sui modelli nuovi, confrontati tra di loro, sul dataset di test con una confidenza del 20%:

| Modello     |   mAP50 |   mAP50-95 |   Precision |   Recall |
|:------------|--------:|-----------:|------------:|---------:|
| 26s_newData |   0.614 |      0.295 |       0.644 |    0.532 |
| 26m_newData |   0.659 |      0.306 |       0.703 |    0.571 |
| 26l_newData |   0.662 |      0.317 |       0.703 |    0.579 |

Confrontando il modello addestrato con il dataset aggiornato notiam un leggero peggioramento rispetto al precedente:

| Modello     |   mAP50 |   mAP50-95 |   Precision |   Recall |
|:------------|--------:|-----------:|------------:|---------:|
| 26l         |   0.719 |      0.338 |       0.727 |    0.644 |
| 26l_newData |   0.662 |      0.317 |       0.703 |    0.579 |

### Spiegazione delle metriche

Le metriche utilizzate rappresentano:

- **mAP50**, *mean Average Precision 50*, è la precisione media calcolata con una soglia di sovrapposizione del 50% tra la predizione del modello e l'oggetto reale. Ci dice se il sistema è in grado di visualizzare l'oggetto cercato.

- **mAP50-95**, è la media della precisione calcolata su diverse soglie di rigore che vanno dal 50% al 95%. Ci dice quanto è bravo il modello nel predire i contorni dell'oggetto.

- **Precision**, indica la capacità di non generare falsi positivi, è data dal rapporto:  $$\frac{True Positives}{True Positives + False Positives} $$

- **Recall**, indica la capacità di individuare tutti gli oggetti presenti, è dato dal rapporto:
 $$\frac{True Positives}{True Positives + False Negatives} $$

Il modello **Large** risulta essere il migliore tra i nuovi modelli addestrati, con prestazioni stabili su entrambe le metriche di bounding box e mask segmentation.

### Dataset

Per l'adattamento del dataset di addestramento [datasetAggiornato](datasetAggiornato) è stato utilizzato [roboflow](https://roboflow.com/).
