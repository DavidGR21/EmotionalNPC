from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from genetic.genome import generate_random_genome


POPULATION_SIZE = 150
IMAGE_MAX_ROWS = 30


def build_population_dataframe(population_size):
    rows = []
    for idx in range(population_size):
        genome = generate_random_genome()
        genome_with_id = {"individuo": f"ind_{idx + 1:03d}"}
        genome_with_id.update(genome)
        rows.append(genome_with_id)
    return pd.DataFrame(rows)


def save_excel(df, path):
    df.to_excel(path, index=False, sheet_name="poblacion_inicial")


def save_table_image(df, path, max_rows):
    preview = df.head(max_rows).copy()
    preview = preview.round(4)

    n_rows, n_cols = preview.shape
    fig_w = max(14, n_cols * 0.9)
    fig_h = max(8, n_rows * 0.33)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)

    title = f"Matriz AG: primeros {len(preview)} individuos de {len(df)}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    root = ROOT_DIR
    out_dir = root / "data" / "ga_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = build_population_dataframe(POPULATION_SIZE)

    excel_path = out_dir / f"ga_population_matrix_{timestamp}.xlsx"
    image_path = out_dir / f"ga_population_matrix_{timestamp}.png"
    latest_excel = out_dir / "ga_population_matrix_latest.xlsx"
    latest_image = out_dir / "ga_population_matrix_latest.png"

    save_excel(df, excel_path)
    save_table_image(df, image_path, IMAGE_MAX_ROWS)

    save_excel(df, latest_excel)
    save_table_image(df, latest_image, IMAGE_MAX_ROWS)

    print(f"Saved: {excel_path}")
    print(f"Saved: {image_path}")
    print(f"Saved: {latest_excel}")
    print(f"Saved: {latest_image}")


if __name__ == "__main__":
    main()
