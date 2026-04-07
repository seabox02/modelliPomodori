"""
annotate_targets.py — Annotazione umana dei pomodori più raggiungibili.

Per ogni immagine del test set, l'annotatore clicca sui 3 pomodori che
riterrebbe più adatti alla raccolta robotica (in ordine di preferenza).
I click vengono salvati in un file JSON con le coordinate e l'ordine.

Uso:
    python annotate_targets.py --images_dir path/to/test/images --output annotations.json

Comandi durante l'annotazione:
    - Click sinistro: seleziona un pomodoro (max 3 per immagine)
    - 'z': annulla l'ultimo click
    - 'n' o Invio: passa all'immagine successiva (anche con meno di 3 click)
    - 's': salta l'immagine (scene senza pomodori raggiungibili)
    - 'q': salva e esci (puoi riprendere dopo)
"""

import argparse
import json
import os
import glob
import cv2
import numpy as np
from pathlib import Path


class AnnotationTool:
    def __init__(self, images_dir, output_file, annotator_name="default"):
        self.images_dir = images_dir
        self.output_file = output_file
        self.annotator_name = annotator_name
        self.max_clicks = 3

        # Carica immagini
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths.sort()

        if not self.image_paths:
            raise FileNotFoundError(f"Nessuna immagine trovata in {images_dir}")

        # Carica annotazioni esistenti (per riprendere il lavoro)
        self.annotations = self._load_existing()

        # Stato corrente
        self.current_clicks = []
        self.current_image = None
        self.display_image = None
        self.window_name = "Annotazione Raggiungibilita - Click sui 3 pomodori piu accessibili"

    def _load_existing(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                data = json.load(f)
            print(f"Caricate {len(data.get('annotations', {}))} annotazioni esistenti.")
            return data
        return {
            "annotator": self.annotator_name,
            "images_dir": self.images_dir,
            "max_targets": self.max_clicks,
            "tolerance_px": 60,
            "annotations": {}
        }

    def _save(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.current_clicks) >= self.max_clicks:
            print(f"  Hai gia selezionato {self.max_clicks} pomodori. Premi 'z' per annullare o 'n' per proseguire.")
            return

        self.current_clicks.append((x, y))
        rank = len(self.current_clicks)
        print(f"  Click #{rank}: ({x}, {y})")

        self._redraw()

    def _redraw(self):
        self.display_image = self.current_image.copy()

        colors = [
            (0, 255, 0),    # verde — primo (più raggiungibile)
            (0, 200, 255),  # arancione
            (0, 0, 255),    # rosso — terzo
        ]

        for i, (cx, cy) in enumerate(self.current_clicks):
            color = colors[i] if i < len(colors) else (255, 255, 255)
            rank = i + 1

            # Cerchio con raggio di tolleranza (60px)
            cv2.circle(self.display_image, (cx, cy), 60, color, 2)
            # Punto centrale
            cv2.circle(self.display_image, (cx, cy), 5, color, -1)
            # Numero del rank
            cv2.putText(self.display_image, str(rank), (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Istruzioni in alto
        h, w = self.display_image.shape[:2]
        info = f"Selezionati: {len(self.current_clicks)}/{self.max_clicks} | 'z'=annulla | 'n'=avanti | 's'=salta | 'q'=esci"
        cv2.rectangle(self.display_image, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(self.display_image, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(self.window_name, self.display_image)

    def run(self):
        # Filtra le immagini già annotate
        remaining = [p for p in self.image_paths
                     if Path(p).name not in self.annotations.get("annotations", {})]

        total = len(self.image_paths)
        done = total - len(remaining)
        print(f"\n{'='*60}")
        print(f"Immagini totali: {total}")
        print(f"Gia annotate: {done}")
        print(f"Rimanenti: {len(remaining)}")
        print(f"{'='*60}\n")

        if not remaining:
            print("Tutte le immagini sono state annotate.")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        for idx, img_path in enumerate(remaining):
            img_name = Path(img_path).name
            print(f"\n[{done + idx + 1}/{total}] {img_name}")

            self.current_image = cv2.imread(img_path)
            if self.current_image is None:
                print(f"  Errore nel caricamento, salto.")
                continue

            # Ridimensiona per lo schermo se troppo grande
            h, w = self.current_image.shape[:2]
            max_dim = 1400
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                self.current_image = cv2.resize(self.current_image,
                                                 (int(w * scale), int(h * scale)))
                self._scale = scale
            else:
                self._scale = 1.0

            self.current_clicks = []
            self._redraw()

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    # Salva e esci
                    self._save()
                    cv2.destroyAllWindows()
                    print(f"\nSalvate {len(self.annotations['annotations'])} annotazioni in {self.output_file}")
                    return

                elif key == ord('z'):
                    # Annulla ultimo click
                    if self.current_clicks:
                        removed = self.current_clicks.pop()
                        print(f"  Annullato click: {removed}")
                        self._redraw()

                elif key == ord('s'):
                    # Salta immagine
                    self.annotations["annotations"][img_name] = {
                        "skipped": True,
                        "reason": "no_reachable_targets"
                    }
                    print(f"  Immagine saltata.")
                    break

                elif key == ord('n') or key == 13:  # 'n' o Invio
                    # Salva e passa alla successiva
                    if not self.current_clicks:
                        print("  Nessun click registrato. Premi 's' per saltare o clicca un pomodoro.")
                        continue

                    # Riconverti le coordinate alla risoluzione originale
                    original_clicks = []
                    for (cx, cy) in self.current_clicks:
                        ox = int(cx / self._scale)
                        oy = int(cy / self._scale)
                        original_clicks.append({"x": ox, "y": oy})

                    self.annotations["annotations"][img_name] = {
                        "skipped": False,
                        "targets": original_clicks
                    }
                    print(f"  Salvati {len(original_clicks)} target.")
                    break

        self._save()
        cv2.destroyAllWindows()
        print(f"\nAnnotazione completata. Salvate {len(self.annotations['annotations'])} annotazioni in {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotazione manuale dei pomodori raggiungibili")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Cartella con le immagini del test set")
    parser.add_argument("--output", type=str, default="annotations_reachability.json",
                        help="File JSON di output")
    parser.add_argument("--annotator", type=str, default="annotatore_1",
                        help="Nome dell'annotatore (per distinguere più annotatori)")
    args = parser.parse_args()

    tool = AnnotationTool(args.images_dir, args.output, args.annotator)
    tool.run()
