"""
compare_results.py — Confronto tra ranking dell'algoritmo e annotazioni umane.

Carica le annotazioni umane (da annotate_targets.py) e i risultati
dell'algoritmo di raggiungibilità, poi calcola il tasso di concordanza.

Uso:
    python compare_results.py \
        --human annotations_reachability.json \
        --algorithm algorithm_results.json \
        --images_dir path/to/test/images \
        --output comparison_report.json \
        --visualize

Il file algorithm_results.json deve avere questa struttura:
{
    "image_name.jpg": {
        "targets": [
            {"x": 320, "y": 240, "rank": 1, "area": 0.05, "circularity": 0.92, "mature": true},
            {"x": 510, "y": 180, "rank": 2, ...},
            {"x": 150, "y": 400, "rank": 3, ...}
        ]
    }
}

Se non hai ancora questo file, lo script può generarlo eseguendo
l'algoritmo di raggiungibilità direttamente sulle predizioni del modello.
Vedi l'opzione --run_algorithm.
"""

import argparse
import json
import os
import math
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def distance(p1, p2):
    """Distanza euclidea tra due punti (dict con 'x' e 'y')."""
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)


def compute_agreement(human_targets, algo_targets, tolerance=60):
    """
    Calcola la concordanza tra selezione umana e algoritmica.

    Returns:
        dict con:
        - matched: lista di coppie (indice_umano, indice_algo) matchate
        - match_count: numero di match entro la tolleranza
        - total_human: numero di target umani
        - total_algo: numero di target algoritmici
        - order_agreement: True se i match rispettano l'ordine di ranking
    """
    if not human_targets or not algo_targets:
        return {
            "matched": [],
            "match_count": 0,
            "total_human": len(human_targets) if human_targets else 0,
            "total_algo": len(algo_targets) if algo_targets else 0,
            "order_agreement": False
        }

    # Matching greedy: per ogni target umano, trova il target algoritmico
    # più vicino entro la tolleranza (senza riutilizzare target già matchati)
    used_algo = set()
    matched = []

    for h_idx, h_target in enumerate(human_targets):
        best_dist = float('inf')
        best_a_idx = None

        for a_idx, a_target in enumerate(algo_targets):
            if a_idx in used_algo:
                continue
            d = distance(h_target, a_target)
            if d < best_dist and d <= tolerance:
                best_dist = d
                best_a_idx = a_idx

        if best_a_idx is not None:
            matched.append((h_idx, best_a_idx, best_dist))
            used_algo.add(best_a_idx)

    # Verifica ordine: i match rispettano l'ordine di ranking?
    order_ok = True
    if len(matched) >= 2:
        for i in range(len(matched) - 1):
            if matched[i][1] > matched[i+1][1]:
                order_ok = False
                break

    return {
        "matched": [(h, a) for h, a, _ in matched],
        "distances": [d for _, _, d in matched],
        "match_count": len(matched),
        "total_human": len(human_targets),
        "total_algo": len(algo_targets),
        "order_agreement": order_ok
    }


def generate_report(human_data, algo_data, tolerance=60):
    """Genera il report completo di confronto."""
    human_annotations = human_data.get("annotations", {})
    results = {}
    summary = {
        "total_images": 0,
        "skipped_images": 0,
        "evaluated_images": 0,
        "total_matches": 0,
        "total_human_targets": 0,
        "total_algo_targets": 0,
        "exact_order_matches": 0,
        "per_rank_accuracy": defaultdict(lambda: {"correct": 0, "total": 0}),
        "match_distances": []
    }

    for img_name, human_ann in human_annotations.items():
        summary["total_images"] += 1

        if human_ann.get("skipped", False):
            summary["skipped_images"] += 1
            continue

        if img_name not in algo_data:
            print(f"  Attenzione: {img_name} non trovata nei risultati dell'algoritmo, salto.")
            continue

        summary["evaluated_images"] += 1

        human_targets = human_ann.get("targets", [])
        algo_targets = algo_data[img_name].get("targets", [])

        agreement = compute_agreement(human_targets, algo_targets, tolerance)

        summary["total_matches"] += agreement["match_count"]
        summary["total_human_targets"] += agreement["total_human"]
        summary["total_algo_targets"] += agreement["total_algo"]
        summary["match_distances"].extend(agreement.get("distances", []))

        if agreement["order_agreement"] and agreement["match_count"] == agreement["total_human"]:
            summary["exact_order_matches"] += 1

        # Accuratezza per rank
        for h_idx, a_idx in agreement["matched"]:
            rank = h_idx + 1  # 1-indexed
            summary["per_rank_accuracy"][rank]["total"] += 1
            if h_idx == a_idx:  # stesso rank
                summary["per_rank_accuracy"][rank]["correct"] += 1

        results[img_name] = {
            "match_count": agreement["match_count"],
            "total_human": agreement["total_human"],
            "total_algo": agreement["total_algo"],
            "order_agreement": agreement["order_agreement"],
            "matched_pairs": agreement["matched"],
            "distances": agreement.get("distances", [])
        }

    # Controlla falsi positivi sulle immagini skippate
    false_positive_images = 0
    total_false_positive_targets = 0
    for img_name, human_ann in human_annotations.items():
        if not human_ann.get("skipped", False):
            continue
        if img_name in algo_data:
            algo_targets = algo_data[img_name].get("targets", [])
        if len(algo_targets) > 0:
            false_positive_images += 1
            total_false_positive_targets += len(algo_targets)
            results[img_name] = {
                "false_positive": True,
                "algo_targets": len(algo_targets),
            }

    # Calcola metriche aggregate
    n_eval = summary["evaluated_images"]
    n_human = summary["total_human_targets"]

    report = {
        "config": {
            "tolerance_px": tolerance,
            "total_images": summary["total_images"],
            "skipped_images": summary["skipped_images"],
            "evaluated_images": n_eval,
        },
        "metrics": {
            "concordance_rate": round(summary["total_matches"] / n_human, 4) if n_human > 0 else 0,
            "exact_order_rate": round(summary["exact_order_matches"] / n_eval, 4) if n_eval > 0 else 0,
            "mean_match_distance_px": round(np.mean(summary["match_distances"]), 2) if summary["match_distances"] else 0,
            "total_matches": summary["total_matches"],
            "total_human_targets": n_human,
        },
        "per_rank": {},
        "per_image": results
    }

    report["false_positives"] = {
        "skipped_images_with_algo_targets": false_positive_images,
        "total_skipped": summary["skipped_images"],
        "total_false_targets": total_false_positive_targets,
    }

    for rank in sorted(summary["per_rank_accuracy"].keys()):
        data = summary["per_rank_accuracy"][rank]
        report["per_rank"][f"rank_{rank}"] = {
            "correct": data["correct"],
            "total": data["total"],
            "accuracy": round(data["correct"] / data["total"], 4) if data["total"] > 0 else 0
        }

    return report


def visualize_comparison(img_path, human_targets, algo_targets, matched_pairs, tolerance=60, save_path=None):
    """Mostra e salva il confronto visivo su un'immagine."""
    img = cv2.imread(img_path)
    if img is None:
        return

    matched_human = {h for h, a in matched_pairs}
    matched_algo = {a for h, a in matched_pairs}

    # Target algoritmici (quadrati)
    for i, t in enumerate(algo_targets):
        color = (255, 150, 0) if i in matched_algo else (150, 150, 150)
        x, y = t["x"], t["y"]
        cv2.rectangle(img, (x-8, y-8), (x+8, y+8), color, 2)
        cv2.putText(img, f"A{i+1}", (x+12, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Target umani (cerchi)
    for i, t in enumerate(human_targets):
        color = (0, 255, 0) if i in matched_human else (0, 0, 255)
        x, y = t["x"], t["y"]
        cv2.circle(img, (x, y), tolerance, color, 2)
        cv2.circle(img, (x, y), 5, color, -1)
        cv2.putText(img, f"H{i+1}", (x+12, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Linee tra i match
    for h_idx, a_idx in matched_pairs:
        h = human_targets[h_idx]
        a = algo_targets[a_idx]
        cv2.line(img, (h["x"], h["y"]), (a["x"], a["y"]), (255, 255, 0), 1)

    # Barra info in alto
    n_match = len(matched_pairs)
    n_total = len(human_targets)
    cv2.rectangle(img, (0, 0), (img.shape[1], 45), (0, 0, 0), -1)
    info = f"Match: {n_match}/{n_total} | Verde=umano Blu=algoritmo Giallo=match"
    cv2.putText(img, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Salva se richiesto
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

    # Mostra a schermo
    cv2.namedWindow("Confronto", cv2.WINDOW_NORMAL)
    cv2.imshow("Confronto", img)
    cv2.waitKey(0)


def print_report(report):
    """Stampa un riepilogo leggibile del report."""
    m = report["metrics"]
    c = report["config"]

    print(f"\n{'='*60}")
    print(f"REPORT CONFRONTO RAGGIUNGIBILITÀ")
    print(f"{'='*60}")
    print(f"Immagini valutate:        {c['evaluated_images']} (su {c['total_images']}, {c['skipped_images']} saltate)")
    print(f"Tolleranza matching:      {c['tolerance_px']} px")
    print(f"")
    print(f"Tasso di concordanza:     {m['concordance_rate']*100:.1f}% ({m['total_matches']}/{m['total_human_targets']})")
    print(f"Concordanza esatta:       {m['exact_order_rate']*100:.1f}% (stesso set E stesso ordine)")
    print(f"Distanza media match:     {m['mean_match_distance_px']:.1f} px")
    print(f"")

    if report["per_rank"]:
        print(f"Accuratezza per rank:")
        for rank_name, data in report["per_rank"].items():
            print(f"  {rank_name}: {data['accuracy']*100:.1f}% ({data['correct']}/{data['total']})")

    if "false_positives" in report:
        fp = report["false_positives"]
        print(f"Immagini senza target con selezioni algoritmo: {fp['skipped_images_with_algo_targets']}/{fp['total_skipped']}")
        print(f"Falsi positivi totali: {fp['total_false_targets']}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confronto annotazioni umane vs algoritmo")
    parser.add_argument("--human", type=str, required=True,
                        help="File JSON con le annotazioni umane (da annotate_targets.py)")
    parser.add_argument("--algorithm", type=str, required=True,
                        help="File JSON con i risultati dell'algoritmo di raggiungibilità")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Cartella immagini (necessaria solo per --visualize)")
    parser.add_argument("--output", type=str, default="comparison_report.json",
                        help="File JSON di output con il report")
    parser.add_argument("--tolerance", type=int, default=60,
                        help="Raggio di tolleranza in pixel (default: 60)")
    parser.add_argument("--visualize", action="store_true",
                        help="Mostra il confronto visivo per ogni immagine")
    args = parser.parse_args()

    human_data = load_json(args.human)
    algo_data = load_json(args.algorithm)

    report = generate_report(human_data, algo_data, args.tolerance)

    # Salva report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print(f"Report completo salvato in {args.output}")

    # Visualizzazione
    if args.visualize and args.images_dir:
        human_annotations = human_data.get("annotations", {})
        for img_name, img_results in report["per_image"].items():
            img_path = os.path.join(args.images_dir, img_name)
            if not os.path.exists(img_path):
                continue

            human_targets = human_annotations[img_name].get("targets", [])
            algo_targets = algo_data[img_name].get("targets", [])

            save_path = os.path.join("confronto visivo", img_name)
            visualize_comparison(
                img_path, human_targets, algo_targets,
                img_results["matched_pairs"], args.tolerance,
                save_path=save_path
            )

    cv2.destroyAllWindows()
