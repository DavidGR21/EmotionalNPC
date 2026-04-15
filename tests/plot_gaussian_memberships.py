import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian(x, center, sigma):
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def load_personalities(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _add_centroid_and_sigma(ax, center: float, sigma: float, color, label_prefix: str = ""):
    """Dibuja el centroide (c) y una indicación visual de sigma (c±σ) en el pico."""
    c = _clip01(center)
    s = max(0.0, float(sigma))
    # Centroide: línea punteada sutil
    ax.axvline(c, color=color, linestyle="--", linewidth=1.0, alpha=0.55)
    # Sigma: barra de error horizontal en el pico (y≈1)
    left = _clip01(c - s)
    right = _clip01(c + s)
    xerr = np.array([[c - left], [right - c]])
    ax.errorbar(
        [c],
        [1.0],
        xerr=xerr,
        fmt="o",
        markersize=3.5,
        color=color,
        ecolor=color,
        elinewidth=1.0,
        capsize=2,
        alpha=0.9,
        label=None,
    )
    # Etiqueta discreta cerca del pico (evita saturar la leyenda)
    if label_prefix:
        ax.annotate(
            f"{label_prefix}c={c:.3f}, σ={s:.3f}",
            xy=(c, 1.0),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=7,
            color=color,
            alpha=0.9,
        )


def plot_memberships_by_personality(personalities, output_path):
    x = np.linspace(0.0, 1.0, 600)
    names = list(personalities.keys())

    fig, axes = plt.subplots(len(names), 2, figsize=(14, 3.2 * len(names)), sharex=True, sharey=True)
    if len(names) == 1:
        axes = np.array([axes])

    for row_idx, name in enumerate(names):
        params = personalities[name]

        a_centers = {
            "low": params["estimulacion_optima_baja"],
            "medium": params["estimulacion_optima_media"],
            "high": params["estimulacion_optima_alta"],
        }
        a_sigma = params["tolerancia_estimulo"]

        v_centers = {
            "negative": params["umbral_bienestar_negativo"],
            "neutral": params["umbral_bienestar_neutral"],
            "positive": params["umbral_bienestar_positivo"],
        }
        v_sigma = params["tolerancia_incomodidad"]

        ax_a = axes[row_idx, 0]
        for label, c in a_centers.items():
            (line,) = ax_a.plot(x, gaussian(x, c, a_sigma), linewidth=2, label=f"{label} (c={c:.3f})")
            _add_centroid_and_sigma(ax_a, center=c, sigma=a_sigma, color=line.get_color(), label_prefix=f"{label}: ")
        ax_a.set_title(f"{name}: Arousal (sigma={a_sigma:.3f})")
        ax_a.set_xlim(0, 1)
        ax_a.set_ylim(0, 1.05)
        ax_a.grid(alpha=0.25)
        ax_a.legend(fontsize=8, loc="upper right")

        ax_v = axes[row_idx, 1]
        for label, c in v_centers.items():
            (line,) = ax_v.plot(x, gaussian(x, c, v_sigma), linewidth=2, label=f"{label} (c={c:.3f})")
            _add_centroid_and_sigma(ax_v, center=c, sigma=v_sigma, color=line.get_color(), label_prefix=f"{label}: ")
        ax_v.set_title(f"{name}: Valence (sigma={v_sigma:.3f})")
        ax_v.set_xlim(0, 1)
        ax_v.set_ylim(0, 1.05)
        ax_v.grid(alpha=0.25)
        ax_v.legend(fontsize=8, loc="upper right")

    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("membership")

    fig.suptitle("Gaussian Membership Functions by Personality", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_all_personalities_overlaid(personalities, output_path):
    x = np.linspace(0.0, 1.0, 600)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    for name, params in personalities.items():
        a_sigma = params["tolerancia_estimulo"]
        v_sigma = params["tolerancia_incomodidad"]

        for label, center_key in [
            ("A_low", "estimulacion_optima_baja"),
            ("A_med", "estimulacion_optima_media"),
            ("A_high", "estimulacion_optima_alta"),
        ]:
            c = params[center_key]
            (line,) = axes[0].plot(x, gaussian(x, c, a_sigma), linewidth=1.3, alpha=0.8, label=f"{name}:{label}")
            # En overlay, solo marcamos el centroide (sin texto) para evitar saturación.
            axes[0].plot([_clip01(c)], [1.0], marker="o", markersize=2.5, color=line.get_color(), alpha=0.65)

        for label, center_key in [
            ("V_neg", "umbral_bienestar_negativo"),
            ("V_neu", "umbral_bienestar_neutral"),
            ("V_pos", "umbral_bienestar_positivo"),
        ]:
            c = params[center_key]
            (line,) = axes[1].plot(x, gaussian(x, c, v_sigma), linewidth=1.3, alpha=0.8, label=f"{name}:{label}")
            axes[1].plot([_clip01(c)], [1.0], marker="o", markersize=2.5, color=line.get_color(), alpha=0.65)

    axes[0].set_title("Overlay Arousal Memberships")
    axes[1].set_title("Overlay Valence Memberships")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("x")
        ax.set_ylabel("membership")
        ax.grid(alpha=0.25)

    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")

    fig.suptitle("All Personalities - Gaussian Membership Overlays", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "personalities.json"
    out_dir = root / "data" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    personalities = load_personalities(data_path)

    path_grid = out_dir / "gaussians_by_personality_grid.png"
    path_overlay = out_dir / "gaussians_overlay_all_personalities.png"

    plot_memberships_by_personality(personalities, path_grid)
    plot_all_personalities_overlaid(personalities, path_overlay)

    print(f"Saved: {path_grid}")
    print(f"Saved: {path_overlay}")


if __name__ == "__main__":
    main()
