import json
import os
import time
from copy import deepcopy

from genetic.fitness import get_fitness_evaluator
from train_ga import train_genetic_algorithm

# ==============================
# CONFIGURACION EXPERIMENTAL
# Ajusta estos valores cuando quieras escalar entrenamiento.
# ==============================
PERSONALITIES = ["normal", "cobarde", "valiente", "explorador", "superviviente"]
BEST_OF_N = 5
POPULATION_SIZE = 80
GENERATIONS = 50
MUTATION_RATE = 0.10
ELITISM = 5


def _load_db(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_db(json_path, db):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)


def _run_best_of_n_for_personality(personality, db):
    evaluator = get_fitness_evaluator(personality)

    baseline_genome = db.get(personality, {})
    baseline_score = (
        evaluator(baseline_genome, generation=GENERATIONS, total_generations=GENERATIONS)
        if baseline_genome
        else float("-inf")
    )

    best_genome = deepcopy(baseline_genome)
    best_score = baseline_score
    run_scores = []

    print(f"\n[{personality.upper()}] baseline={baseline_score:.6f}")

    for run_idx in range(1, BEST_OF_N + 1):
        candidate = train_genetic_algorithm(
            personality_type=personality,
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            mutation_rate=MUTATION_RATE,
            elitism=ELITISM,
        )
        score = evaluator(candidate, generation=GENERATIONS, total_generations=GENERATIONS)
        run_scores.append(score)
        print(f"  run{run_idx}: {score:.6f}")

        if score > best_score:
            best_score = score
            best_genome = deepcopy(candidate)

    db[personality] = best_genome

    delta = best_score - baseline_score
    print(f"[{personality.upper()}] selected={best_score:.6f} delta={delta:.6f}")

    return {
        "personality": personality,
        "baseline_score": baseline_score,
        "selected_score": best_score,
        "delta": delta,
        "run_scores": run_scores,
    }


def main():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(root_dir, "data", "personalities.json")

    report_dir = os.path.join(root_dir, "tests", "training_reports")
    os.makedirs(report_dir, exist_ok=True)

    db = _load_db(json_path)
    report_items = []

    print("=== BEST-OF-N TRAINING ===")
    print(
        f"Config -> best_of={BEST_OF_N}, population={POPULATION_SIZE}, "
        f"generations={GENERATIONS}, mutation_rate={MUTATION_RATE}, elitism={ELITISM}"
    )

    for personality in PERSONALITIES:
        result = _run_best_of_n_for_personality(personality, db)
        report_items.append(result)
        _save_db(json_path, db)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_payload = {
        "timestamp": timestamp,
        "config": {
            "best_of_n": BEST_OF_N,
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "mutation_rate": MUTATION_RATE,
            "elitism": ELITISM,
            "personalities": PERSONALITIES,
        },
        "results": report_items,
    }

    report_json_path = os.path.join(report_dir, f"best_of_n_{timestamp}.json")
    latest_json_path = os.path.join(report_dir, "latest_report.json")

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4)

    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4)

    print("\n=== TRAINING REPORT SAVED ===")
    print(f"Report: {report_json_path}")
    print(f"Latest: {latest_json_path}")


if __name__ == "__main__":
    main()
