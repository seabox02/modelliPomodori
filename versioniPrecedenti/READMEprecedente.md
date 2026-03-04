## Modelli object detection e mask segmentation su dataset pomodori 

Sviluppati nuovi modelli su più epoche (100/150) e usando versioni di yolo più potenti (come large e extralarge). 
I primi modelli si possono trovare [qui](runs/segment0). (anche se sono peggiori rispetto a quelli nuovi)

Oltre alle varie versioni dei modelli abbiamo tenuto conto del _batch size_, che ha un impatto importante sulle metriche di precisione.

In questo grafico vediamo 3 modelli addestrati su diverse versioni di YOLO11, rispettivamente medium, large ed extralarge.  
Il quarto modello invece è una versione _ibrida_ in cui viene modificata l'architettura della versione medium. La modifica è stata effettuata a livello di _backbone_, inserendo un modulo di attenzione C2PSA, che teoricamente avrebbe dovuto far diminuire il numero di falsi positivi ed aumentare la precisione soprattutto a livello di mask-segmentation. In realtà non ha funzionato come previsto, ma potrebbe essere un'idea da sviluppare meglio. Il modello ibrido e l'extralarge sono addestrati con batch size di 32, medium e large con batch size di 64. 

![Grafico Evoluzione Prestazioni](evolution_chart11_32batch.png)

Abbiamo modelli della stessa versione di YOLO, ad esempio YOLO11-medium, addestrati su diverse scelte di batch (16, 32 e 64).

Possiamo notare nel grafico qui sotto come la precisione (mAP) di modelli addestrati con YOLO26-medium cambi a seconda del batch_size scelto. 

![Grafico evoluzione validazione cambiando batch](evolution_chart_medium26.png)

Qui vediamo un confronto quantitativo su diverse metriche, tra i 3 modelli qui sopra, utilizzando il modello sul dataset di test, ottenendo così stime sensate, in quanto
lavora su immagini che non aveva mai visto prima.

Qui visualizziamo le metriche riguardo le bounding box:

| Modello     |   mAP50 |   mAP50-95 |   Precision |   Recall |   
|:------------|--------:|-----------:|------------:|---------:|
| YOLO26m_B16 | *0.808* |      0.498 |     *0.827* |    0.737 |
| YOLO26m_B32 |   0.793 |      0.509 |       0.78  | *0.740*  |
| YOLO26m_B64 |   0.798 |    *0.512* |       0.804 |    0.711 |

In questa tabella invece abbiamo le metriche riguardo le maschere di segmentazione:

| Modello     |   mAP50 |   mAP50-95 |   Precision |   Recall |
|:------------|--------:|-----------:|------------:|---------:|
| YOLO26m_B16 |   0.669 |      0.293 |     *0.732* |  *0.671* |
| YOLO26m_B32 | *0.672* |    *0.295* |       0.708 |    0.67  |
| YOLO26m_B64 |   0.667 |      0.29  |       0.722 |    0.634 |

Notiamo che in realtà, che non c'è una grande variazione tra i modelli della stessa versione usando un batch differente.

Le metriche utilizzate rappresentano:
 - mAP50, _mean Average Precision 50_, è la precisione media calcolata con una soglia di sovrapposizine del 50% tra la predizione del modello e l'oggetto reale. Ci dice se il sistema è in grade di visualizzare l'oggetto cercato.
 - mAPA50-95, è la media della precisione calcolata su diverse soglie di rigore che vanno dal 50% al 95%. Ci dice quanto è bravo il modello nel predire i contorni dell'oggetto.
 - Precision, indica la capacità di non generare falsi positivi, è data dal rapporto:  $$\frac{True Positives}{True Positives + False Positives} $$
 - Recall, indica la capacità di individuare tutti gli oggetti presenti, è dato dal rapporto:
 $$\frac{True Positives}{True Positives + False Negatives} $$

Qui vediamo invece come varia la precisione dei modelli addestrati con YOLO26-large

| Modello     |   mAP50 |   mAP50-95 |   Precision |   Recall |
|:------------|--------:|-----------:|------------:|---------:|
| YOLO26l_B16 |   0.675 |      0.291 |       0.715 |    0.663 |
| YOLO26l_B32 |   0.69  |      0.288 |       0.728 |    0.676 |
| YOLO26l_B64 |   0.702 |      0.299 |       0.72  |    0.674 |

Da queste tabelle possiamo notare che tra la versione large e medium non cambia moltissimo riguardo queste metriche, quello che comunque sembra essere il migliore qui è la versione large adestrata con 64 batch.  

Possiamo notare che questo modello sembra essere migliore anche della versione extralarge (32 batch)

| Modello     |   mAP50 |   mAP50-95 |   Precision |   Recall |
|:------------|--------:|-----------:|------------:|---------:|
| YOLO26l_B64 |   0.702 |      0.3   |       0.72  |    0.674 |
| YOLO26x_B32 |   0.677 |      0.305 |       0.698 |    0.674 |

In questa tabella invece confrontiamo la versione migliore di YOLO11 e la versione migliore di YOLO26. Nonostante il medium con yolo11 abbia una precisione migliore rispetto alla versione large di yolo26, tutte le altre metriche sono migliori. 
Possiamo dire quindi che il modello migliore sembra essere quello addestrato con la _versione large di YOLO26_.

| Modello     |   mAP50 |   mAP50-95 |   Precision |   Recall |
|:------------|--------:|-----------:|------------:|---------:|
| YOLO26l_B64 |   0.702 |      0.3   |       0.72  |    0.674 |
| YOLO11m_B64 |   0.68  |      0.303 |       0.787 |    0.619 |


Per l'adattamento del dataset di addestramento [datasetPomdori](datasetPomodori) è stato utilizzato [roboflow](https://roboflow.com/).
