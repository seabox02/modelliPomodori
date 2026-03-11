import pandas as pd
import matplotlib.pyplot as plt
import os

# Configurazione stile professionale
plt.style.use('seaborn-v0_8-whitegrid')
rcParams = {
    'figure.figsize': (12, 8), 
    'axes.titlesize': 14, 
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.8
}
plt.rcParams.update(rcParams)

def load_results(path, name):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    return None

# Caricamento modelli
models = {
    'Small V1 (640px)': 'runs/segment/small26_200_32/results.csv',
    'Small V2 (800px)': 'runs/segment/s26Finale/results.csv',
    'Medium (800px)': 'runs/segment/m26Finale/results.csv',
    'Large V1 (640px)': 'runs/segment/large26_200_32/results.csv',
    'Large V2 Best (800px)': 'runs/segment/Nuovol26/results.csv',
    'ExtraLarge (800px)': 'runs/segment/NuovoXL/results.csv'
}

dataframes = {}
efficiency_data = []
weights = {
    'Small V1 (640px)': 22, 'Small V2 (800px)': 22,
    'Medium (800px)': 52, 'Large V1 (640px)': 61,
    'Large V2 Best (800px)': 61, 'ExtraLarge (800px)': 135
}

for name, path in models.items():
    df = load_results(path, name)
    if df is not None:
        dataframes[name] = df
        last_row = df.iloc[-1]
        efficiency_data.append({
            'Model': name, 'Time_Hours': last_row['time'] / 3600,
            'mAP95_M': last_row['metrics/mAP50-95(M)'], 'Weight_MB': weights.get(name, 0)
        })

def plot_smart_metric(dfs, metric, title, filename, ylabel='mAP@.50:.95', legend_loc='lower right'):
    plt.figure(figsize=(12, 7))
    for name, df in dfs.items():
        if metric in df.columns:
            plt.plot(df['epoch'], df[metric], label=name, linewidth=2.3)
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Epoca')
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# --- GENERAZIONE GRAFICI ---

if dataframes:
    plot_smart_metric(dataframes, 'metrics/mAP50-95(B)', 'Precisione Bounding Box (V1 vs V2)', 'confronto_bbox_95.png', legend_loc='lower right')
    plot_smart_metric(dataframes, 'metrics/mAP50-95(M)', 'Precisione Maschere (V1 vs V2)', 'confronto_mask_95.png', legend_loc='lower right')
    plot_smart_metric(dataframes, 'val/seg_loss', 'Analisi Convergenza: Validation Segmentation Loss', 'analisi_overfitting_seg.png', ylabel='Loss', legend_loc='upper right')

    # Grafico Efficienza CORRETTO (Senza pallina fantasma nella legenda)
    eff_df = pd.DataFrame(efficiency_data)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Disegniamo i punti
    scatter = ax.scatter(eff_df['Time_Hours'], eff_df['mAP95_M'], s=eff_df['Weight_MB']*18, 
                        alpha=0.6, c=range(len(eff_df)), cmap='viridis', edgecolors='black')
    
    # Aggiungiamo le etichette con un offset per evitare sovrapposizioni
    for i, row in eff_df.iterrows():
        ax.annotate(row['Model'], (row['Time_Hours'], row['mAP95_M']), 
                    xytext=(12, 5), textcoords='offset points', 
                    fontsize=9, fontweight='bold')

    ax.set_title('Pareto Frontier: Precisione vs Tempo di Training', fontweight='bold', pad=20)
    ax.set_xlabel('Tempo Training (Ore)')
    ax.set_ylabel('mAP50-95 (Mask)')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Nota testuale invece di una legenda che crea icone extra
    plt.figtext(0.15, 0.05, "*Dimensione bolla proporzionale al peso del modello (MB)", 
                fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('pareto_efficiency.png', dpi=300)
    print("Grafico Efficienza pulito e salvato: pareto_efficiency.png")
    plt.show()
