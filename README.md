## Modelli object detection e mask segmentation su dataset pomodori 

Sviluppati 4 modelli tramite varie versioni di yolov11 (nano, small, medium) addestrati su diverse epoche. Abbiamo sviluppato anche un modello ibrido seguendo l'idea sviluppata nel [paper](https://www.mdpi.com/2076-3417/15/5/2718) su AYOLO.

Sono presenti tool di visualizzazione per testare i modelli:
-  [slideshow.py](slideshow.py) permette di visualizzare le varie immagini applicando le maschere di segmentazione su ogni oggetto e le etichette con il punteggio di confidenza;
- [slideshowNoText.py](slideshow_noText.py) permette di visualizzare esclusivamente le maschere colorate sugli oggetti.

Si può notare nel grafico qui sotto che il modello migliore è sicuramente quello ottenuto lavorando sulla versione _medium_ addestrata su 50 epoche, che raggiunge una precisione circa del 72,1% sul dataset dei pomodori. Risulta migliore anche del modello _nano_ addestrato su più epoche (75); inoltre, possiamo notare che anche rispetto al modello _ibrido_ è di gran lunga migliore nonostante utilizzi la versione medium di yolov11 come "base".

![Grafico Evoluzione Prestazioni](evolution_chart.png)

Oltre al dataset di addestramento, è possibile testare i vari modelli su un ulteriore [dataset](https://datasetninja.com/tomatod#images) . Si può notare come il modello reagisca correttamente anche ad immagini di altre serre, che non aveva mai visto prima.
Nella cartella è [datasetTest](datasetTest)



Per l'adattamento del dataset di addestramento [datasetPomdori](datasetPomodori) è stato utilizzato [roboflow](https://roboflow.com/).