import pandas as pd
import matplotlib.pyplot as plt
import os

files11 = {
    '1. Medium (100 ep)': 'runs/segment11/modello11_medium_100/results.csv',
    '2. Large (100 ep)': 'runs/segment11/modello11_large_100/results.csv',
    '3. ExtraLarge (150 ep)': 'runs/segment11/modello11_extralarge_150/results.csv',
    '4. Ibrido (150 ep)': 'runs/segment11/ibrido11_med_attenzione_150/results.csv',
}

files26 = {
    '1. Medium (150 ep)': 'runs/segment26/modello26_medium_150/results.csv',
    '2. Large (150 ep)': 'runs/segment26/modello26_large_150/results.csv',
    '3. ExtraLarge (150 ep)': 'runs/segment26/modello26_extraLarge_150/results.csv'
}

files_medium_26 = {
    '1. 16 batch (150 ep)': 'runs/segment26/medium/modello26_medium_150/results.csv',
    '2. 32 batch (150 ep)': 'runs/segment26/medium/modello26_medium_150_32/results.csv',
    '3. 64 batch (150 ep)': 'runs/segment26/medium/modello26_medium_150_64/results.csv'
}

dataframes = {}

plt.figure(figsize=(12, 6))

colors11 = ['blue', 'orange', 'purple', 'green']
colors26 = ['blue', 'orange', 'purple']

for i, (model_name, file_path) in enumerate(files_medium_26.items()): # modifica qui per scegliere che modelli visualizzare (files11 o files26)
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.columns = [c.strip() for c in df.columns]
            
            if 'metrics/mAP50(M)' in df.columns:
                col_map = 'metrics/mAP50(M)'
            else:
                continue 

            plt.plot(df[col_map], label=model_name, color=colors26[i], linestyle='-', linewidth=1.5) # modifica anche qui i colori come sopra
            
        else:
            print(f"File non trovato: {file_path}")
            
    except Exception as e:
        print(f"Errore con {model_name}: {e}")

plt.title('Evoluzione train modelli YOLO26-medium', fontsize=14)
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Precisione (mAP50)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvo l'immagine
output_filename = 'evolution_chart_medium26.png'
plt.savefig(output_filename, dpi=300)

print(f"Grafico salvato come {output_filename}")