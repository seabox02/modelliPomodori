import pandas as pd
import matplotlib.pyplot as plt
import os

# Definisco i percorsi dei 4 file rappresentativi della "Storia" del progetto
files = {
    '1. Medium (100 ep)': 'runs/segment/modello_medium_100/results.csv',
    '2. Large (100 ep)': 'runs/segment/modello_large_100/results.csv',
    '3. Extra large 11 (150 ep)': 'runs/segment/modello_extralarge_150/results.csv',
}

dataframes = {}

plt.figure(figsize=(12, 6))

colors = ['blue', 'orange', 'purple']
styles = [':',  '-', ":"] 

# Preparo i dati
for i, (model_name, file_path) in enumerate(files.items()):
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.columns = [c.strip() for c in df.columns]
            
            if 'metrics/mAP50(M)' in df.columns:
                col_map = 'metrics/mAP50(M)'
            elif 'metrics/mAP50(B)' in df.columns:
                col_map = 'metrics/mAP50(B)'
            else:
                continue 

            plt.plot(df[col_map], label=model_name, color=colors[i], linestyle=styles[i], linewidth=2 if i==3 else 1.5)
            
        else:
            print(f"File non trovato: {file_path}")
            
    except Exception as e:
        print(f"Errore con {model_name}: {e}")

plt.title('Evoluzione delle prestazioni: Da Nano a Medium', fontsize=14)
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Precisione (mAP50)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvo l'immagine
output_filename = 'evolution_chart.png'
plt.savefig(output_filename, dpi=300)

print(f"Grafico salvato come {output_filename}")