"""Plot GA training evolution (fitness vs generation).

This script parses the console logs produced by `train_ga.py` / `tests/train_best_of_n.py`
(typically redirected to `train_output.txt`) and generates a figure showing how fitness
improves across generations.

Expected log patterns:
- "=== Iniciando Entrenamiento Evolutivo para: <PERSONALITY> ==="
- "Gen XX | Mejor Score:  124.0 | Media:   87.3"

Output:
- tests/plots/ga_evolution_fitness.png (by default)

Usage:
    py -m tests.plot_ga_evolution
    py tests/plot_ga_evolution.py --log train_output.txt --out tests/plots/ga_evolution_fitness.png
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunSeries:
    personality: str
    generations: List[int]
    best_scores: List[float]
    avg_scores: List[float]


@dataclass
class PlotContext:
    title: str
    footer_lines: List[str]


_HEADER_RE = re.compile(
    r"^===\s*Iniciando\s+Entrenamiento\s+Evolutivo\s+para:\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ_\-]+)\s*===\s*$",
    re.IGNORECASE,
)

_GEN_RE = re.compile(
    r"^Gen\s+(\d+)\s*\|\s*Mejor\s+Score:\s*([-+]?\d+(?:\.\d+)?)\s*\|\s*Media:\s*([-+]?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)

_MD_BLX_ALPHA_RE = re.compile(r"BLX-\s*α.*?α\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_MD_MUT_SIGMA_SCALE_RE = re.compile(r"\\sigma\s*=\s*([0-9]+(?:\.[0-9]+)?)\\,?\s*\(max-min\)", re.IGNORECASE)
_MD_EXAMPLE_CLI_RE = re.compile(
    r"py\s+train_ga\.py\s+\w+\s+(\d+)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+(\d+)",
    re.IGNORECASE,
)

_MD_SELECTION_RE = re.compile(r"truncation\s+selection|best_half|mitad\s+superior", re.IGNORECASE)
_MD_ELITISM_RE = re.compile(r"\bELITISM\b|elitismo", re.IGNORECASE)
_MD_REPAIR_RE = re.compile(r"reparaci[oó]n.*centros|orden.*centros", re.IGNORECASE)


def _normalize_line(line: str) -> str:
    """Normalize a log line to improve regex matching across encodings."""

    if not line:
        return ""
    # Remove BOMs / zero-width chars / stray NULs
    line = line.replace("\ufeff", "").replace("\u200b", "").replace("\x00", "")
    # Collapse weird whitespace
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _decode_log_bytes(raw: bytes) -> str:
    """Decode log bytes robustly.

    `train_output.txt` is often UTF-16LE when redirected from Windows terminals.
    Sometimes the first BOM byte can be missing or extra bytes appear; we keep
    decoding permissive.
    """

    if not raw:
        return ""

    null_ratio = raw.count(b"\x00") / max(1, len(raw))

    # Heuristic: lots of NUL bytes => UTF-16-ish
    if null_ratio > 0.10:
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(enc, errors="ignore")
            except Exception:
                pass

    # Fallbacks
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc, errors="ignore")
        except Exception:
            pass

    return raw.decode("latin-1", errors="ignore")


def parse_training_log(path: Path) -> List[RunSeries]:
    text = _decode_log_bytes(path.read_bytes())
    raw_lines = text.replace("\r\n", "\n").split("\n")
    lines = [_normalize_line(ln) for ln in raw_lines]
    lines = [ln for ln in lines if ln]

    runs: List[RunSeries] = []
    current: Optional[RunSeries] = None

    for ln in lines:
        m_header = _HEADER_RE.match(ln)
        if m_header:
            # Close previous run
            if current and current.generations:
                runs.append(current)
            personality = m_header.group(1)
            current = RunSeries(personality=personality.lower(), generations=[], best_scores=[], avg_scores=[])
            continue

        m_gen = _GEN_RE.match(ln)
        if m_gen:
            if current is None:
                current = RunSeries(personality="entrenamiento", generations=[], best_scores=[], avg_scores=[])

            gen = int(m_gen.group(1))
            best = float(m_gen.group(2))
            avg = float(m_gen.group(3))

            current.generations.append(gen)
            current.best_scores.append(best)
            current.avg_scores.append(avg)

    if current and current.generations:
        runs.append(current)

    return runs


def _group_runs_by_personality(runs: List[RunSeries]) -> Dict[str, List[RunSeries]]:
    grouped: Dict[str, List[RunSeries]] = {}
    for run in runs:
        grouped.setdefault(run.personality, []).append(run)
    return grouped


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def build_plot_context(
    *,
    md_path: Optional[Path],
    latest_report_path: Optional[Path],
    default_title: str,
) -> PlotContext:
    """Build figure context from INFORME_PROYECTO.md (preferred) and latest_report.json (optional)."""

    footer: List[str] = []

    # ---- From report JSON (actual run config) ----
    if latest_report_path and latest_report_path.exists():
        try:
            payload = json.loads(_safe_read_text(latest_report_path) or "{}")
            cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
            if isinstance(cfg, dict) and cfg:
                parts = []
                if "population_size" in cfg:
                    parts.append(f"población={cfg['population_size']}")
                if "generations" in cfg:
                    parts.append(f"generaciones={cfg['generations']}")
                if "mutation_rate" in cfg:
                    parts.append(f"mutation_rate={cfg['mutation_rate']}")
                if "elitism" in cfg:
                    parts.append(f"elitismo={cfg['elitism']}")
                if "best_of_n" in cfg:
                    parts.append(f"best_of_n={cfg['best_of_n']}")
                if parts:
                    footer.append("Config (reporte): " + ", ".join(parts))
        except Exception:
            pass

    # ---- From report MD (operators + documented constants) ----
    if md_path and md_path.exists():
        md = _safe_read_text(md_path)
        if md:
            m_alpha = _MD_BLX_ALPHA_RE.search(md)
            m_sigma = _MD_MUT_SIGMA_SCALE_RE.search(md)
            op_bits = []
            if m_alpha:
                op_bits.append(f"cruce=BLX-α (α={m_alpha.group(1)})")
            if m_sigma:
                op_bits.append(f"mutación: ruido gaussiano (σ={m_sigma.group(1)}·(max-min))")
            if op_bits:
                footer.append("Operadores (informe): " + ", ".join(op_bits))

            # Selection/repair summary (only if present, to avoid inventing)
            sel_bits = []
            if _MD_SELECTION_RE.search(md):
                sel_bits.append("selección=truncation (top 50%)")
            if _MD_ELITISM_RE.search(md):
                sel_bits.append("elitismo preserva élites")
            if _MD_REPAIR_RE.search(md):
                sel_bits.append("reparación ordena centros")
            if sel_bits:
                footer.append("Mecánica (informe): " + ", ".join(sel_bits))

            m_example = _MD_EXAMPLE_CLI_RE.search(md)
            if m_example:
                pop, gens, mut, elite = m_example.groups()
                footer.append(f"Ejemplo CLI (informe): pop={pop}, gen={gens}, mut={mut}, elit={elite}")

    return PlotContext(title=default_title, footer_lines=footer)


def _append_log_coverage_note(context: PlotContext, runs: List[RunSeries], report_path: Optional[Path]) -> PlotContext:
    """If report says G generations but log has fewer, append a note."""
    if not runs:
        return context

    max_gen = max((max(r.generations) for r in runs if r.generations), default=None)
    if max_gen is None:
        return context

    report_g = None
    if report_path and report_path.exists():
        try:
            payload = json.loads(_safe_read_text(report_path) or "{}")
            cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
            if isinstance(cfg, dict):
                report_g = cfg.get("generations")
        except Exception:
            report_g = None

    if isinstance(report_g, int) and report_g > 0 and max_gen < report_g:
        new_lines = list(context.footer_lines)
        new_lines.append(f"Nota: el log contiene {max_gen}/{report_g} generaciones (posible truncamiento del output).")
        return PlotContext(title=context.title, footer_lines=new_lines)

    return context


def _aggregate_by_generation(series_list: List[RunSeries]) -> Tuple[List[int], List[float], List[float]]:
    """Return (gens, mean, std) for a list of RunSeries with possibly identical generations."""
    if not series_list:
        return [], [], []

    # Collect values by generation
    by_gen: Dict[int, List[float]] = {}
    for s in series_list:
        for g, v in zip(s.generations, s.best_scores):
            by_gen.setdefault(g, []).append(v)

    gens = sorted(by_gen.keys())
    means: List[float] = []
    stds: List[float] = []
    for g in gens:
        vals = by_gen[g]
        mean = sum(vals) / len(vals)
        means.append(mean)
        if len(vals) <= 1:
            stds.append(0.0)
        else:
            var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            stds.append(var ** 0.5)

    return gens, means, stds


def _aggregate_avg_by_generation(series_list: List[RunSeries]) -> Tuple[List[int], List[float], List[float]]:
    if not series_list:
        return [], [], []

    by_gen: Dict[int, List[float]] = {}
    for s in series_list:
        for g, v in zip(s.generations, s.avg_scores):
            by_gen.setdefault(g, []).append(v)

    gens = sorted(by_gen.keys())
    means: List[float] = []
    stds: List[float] = []
    for g in gens:
        vals = by_gen[g]
        mean = sum(vals) / len(vals)
        means.append(mean)
        if len(vals) <= 1:
            stds.append(0.0)
        else:
            var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            stds.append(var ** 0.5)

    return gens, means, stds


def plot_evolution(runs: List[RunSeries], out_path: Path, context: PlotContext) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Professional-ish defaults (no extra dependencies)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    grouped = _group_runs_by_personality(runs)

    fig, (ax_best, ax_avg) = plt.subplots(2, 1, figsize=(13.5, 8.6), sharex=True)

    personalities = sorted(grouped.keys())
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(personalities)}

    # Track global best point across all runs
    global_best: Tuple[float, int, str] = (float("-inf"), -1, "")

    for personality, series_list in sorted(grouped.items(), key=lambda kv: kv[0]):
        color = color_map[personality]

        # Single run: plot directly (no misleading "media")
        if len(series_list) == 1:
            series = series_list[0]
            ax_best.plot(series.generations, series.best_scores, linewidth=2.6, color=color, label=personality)
            ax_avg.plot(series.generations, series.avg_scores, linewidth=2.6, color=color, label=personality)

            if series.best_scores:
                local_max = max(series.best_scores)
                if local_max > global_best[0]:
                    g_idx = int(np.argmax(series.best_scores))
                    global_best = (local_max, series.generations[g_idx], personality)
        else:
            # Multiple runs: plot faint runs + aggregated curve (mean ± std)
            for idx, series in enumerate(series_list, start=1):
                ax_best.plot(
                    series.generations,
                    series.best_scores,
                    linewidth=1.4,
                    alpha=0.25,
                    color=color,
                    label=f"{personality} (runs)" if idx == 1 else None,
                )
                ax_avg.plot(
                    series.generations,
                    series.avg_scores,
                    linewidth=1.4,
                    alpha=0.25,
                    color=color,
                    label=f"{personality} (runs)" if idx == 1 else None,
                )

                if series.best_scores:
                    local_max = max(series.best_scores)
                    if local_max > global_best[0]:
                        g_idx = int(np.argmax(series.best_scores))
                        global_best = (local_max, series.generations[g_idx], personality)

            gens_b, mean_b, std_b = _aggregate_by_generation(series_list)
            gens_a, mean_a, std_a = _aggregate_avg_by_generation(series_list)
            if gens_b:
                ax_best.plot(gens_b, mean_b, linewidth=2.6, color=color, label=f"{personality} (media)")
                if any(s > 0 for s in std_b):
                    lo = [m - s for m, s in zip(mean_b, std_b)]
                    hi = [m + s for m, s in zip(mean_b, std_b)]
                    ax_best.fill_between(gens_b, lo, hi, color=color, alpha=0.10)

            if gens_a:
                ax_avg.plot(gens_a, mean_a, linewidth=2.6, color=color, label=f"{personality} (media)")
                if any(s > 0 for s in std_a):
                    lo = [m - s for m, s in zip(mean_a, std_a)]
                    hi = [m + s for m, s in zip(mean_a, std_a)]
                    ax_avg.fill_between(gens_a, lo, hi, color=color, alpha=0.10)

    ax_best.set_title(context.title)
    ax_best.text(
        0.0,
        1.06,
        "Mejor fitness por generación",
        transform=ax_best.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="0.25",
    )
    ax_best.set_ylabel("Mejor fitness")
    ax_best.grid(True, alpha=0.28)

    ax_avg.text(
        0.0,
        1.06,
        "Fitness promedio por generación",
        transform=ax_avg.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="0.25",
    )

    ax_avg.set_ylabel("Fitness promedio")
    ax_avg.set_xlabel("Generación")
    ax_avg.grid(True, alpha=0.28)

    # Annotate global best
    if global_best[1] >= 0:
        best_val, best_gen, best_p = global_best
        ax_best.scatter([best_gen], [best_val], s=80, marker="*", color="black", zorder=5)
        ax_best.axvline(best_gen, color="0.35", linestyle="--", linewidth=1.2, alpha=0.55)
        ax_avg.axvline(best_gen, color="0.35", linestyle="--", linewidth=1.2, alpha=0.55)
        ax_best.annotate(
            f"Mejor global: {best_val:.2f}\nGen {best_gen} ({best_p})",
            xy=(best_gen, best_val),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 1.0},
        )

        # Simple improvement summary (first generation vs last)
        try:
            # Use the run that achieved the global best for baseline
            target_runs = grouped.get(best_p, [])
            if target_runs:
                s0 = target_runs[0]
                if s0.generations and s0.best_scores:
                    baseline = s0.best_scores[0]
                    delta = best_val - baseline
                    if baseline != 0:
                        pct = (delta / abs(baseline)) * 100
                        txt = f"Δ desde Gen1: {delta:.2f} ({pct:.1f}%)"
                    else:
                        txt = f"Δ desde Gen1: {delta:.2f}"
                    ax_best.text(
                        0.99,
                        0.02,
                        txt,
                        transform=ax_best.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=9,
                        color="0.25",
                        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.85", "alpha": 0.85},
                    )

                    # Plateau / last improvement
                    last_gen = s0.generations[-1]
                    last_improve_gen = s0.generations[0]
                    prev = s0.best_scores[0]
                    for g, v in zip(s0.generations[1:], s0.best_scores[1:]):
                        if v > prev + 1e-9:
                            last_improve_gen = g
                            prev = v
                    plateau_len = max(0, last_gen - last_improve_gen)
                    ax_best.text(
                        0.01,
                        0.02,
                        f"Última mejora: Gen {last_improve_gen} · Plateau: {plateau_len} gens",
                        transform=ax_best.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        color="0.25",
                        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.85", "alpha": 0.85},
                    )
        except Exception:
            pass

    # Legend (deduplicate labels)
    handles, labels = ax_best.get_legend_handles_labels()
    seen = set()
    dedup = [(h, l) for h, l in zip(handles, labels) if (l and (l not in seen) and not seen.add(l))]
    if dedup:
        ax_best.legend(
            [h for h, _ in dedup],
            [l for _, l in dedup],
            loc="upper left",
            ncol=2,
            frameon=True,
            framealpha=0.9,
        )

    # Context panel (inside the bottom axis to avoid cropping)
    if context.footer_lines:
        panel = "\n".join(context.footer_lines)
        ax_avg.text(
            0.01,
            -0.34,
            panel,
            transform=ax_avg.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="0.25",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.85", "alpha": 0.95},
        )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot GA fitness evolution from training logs.")
    parser.add_argument(
        "--log",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "train_output.txt"),
        help="Path to training log (default: ./train_output.txt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "plots" / "ga_evolution_fitness.png"),
        help="Output PNG path (default: tests/plots/ga_evolution_fitness.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Evolución del fitness por generación (Algoritmo Genético)",
        help="Figure title",
    )
    parser.add_argument(
        "--md",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "INFORME_PROYECTO.md"),
        help="Path to INFORME_PROYECTO.md (used to enrich figure annotations)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(Path(__file__).resolve().parent / "training_reports" / "latest_report.json"),
        help="Optional training report JSON to show actual experiment config",
    )

    args = parser.parse_args(argv)

    log_path = Path(args.log)
    out_path = Path(args.out)

    if not log_path.exists():
        raise SystemExit(f"No existe el archivo de log: {log_path}")

    runs = parse_training_log(log_path)
    if not runs:
        raise SystemExit(
            "No se encontraron líneas de entrenamiento. "
            "Asegúrate de que el log contenga 'Gen XX | Mejor Score: ... | Media: ...'."
        )

    md_path = Path(args.md) if args.md else None
    report_path = Path(args.report) if args.report else None
    context = build_plot_context(
        md_path=md_path,
        latest_report_path=report_path,
        default_title=args.title,
    )
    context = _append_log_coverage_note(context, runs, report_path)

    plot_evolution(runs, out_path, context)
    print(f"Saved plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
