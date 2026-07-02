#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_variables_temporales_y_toe_clearance.py

Tercera parte del pipeline para análisis de marcha murina con DeepLabCut.
Este script SOLO calcula variables temporales simples y toe clearance usando las
salidas del script 01.

Entrada esperada:
    *_clean_coords.csv
    *_gait_cycles.csv

Este script hace:
    1. Leer coordenadas limpias del script 01.
    2. Leer ciclos de marcha detectados/validados del script 01.
    3. Calcular tiempo de zancada por ciclo.
    4. Estimar toe-off dentro de cada ciclo.
    5. Calcular porcentaje de apoyo.
    6. Calcular porcentaje de oscilación.
    7. Calcular toe clearance por ciclo en pixeles.
    8. Exportar tabla por ciclo.
    9. Exportar resumen por video: media, D.E. y S.E.M.
   10. Graficar control visual de fases y toe clearance.

Este script NO detecta ciclos nuevos, NO calcula ángulos, NO calcula rangos
angulares y NO hace estadística grupal.

Uso recomendado:
    python 03_variables_temporales_y_toe_clearance.py \
        "salida_01/archivo_clean_coords.csv" \
        --cycles "salida_01/archivo_gait_cycles.csv" \
        --fps 30 \
        --outdir salida_03_temporal

Si el archivo de ciclos está en la misma carpeta y tiene el nombre estándar del
script 01, se puede omitir --cycles:
    python 03_variables_temporales_y_toe_clearance.py "salida_01/archivo_clean_coords.csv"

Definiciones usadas:
    - Tiempo de zancada: tiempo entre start_frame y end_frame del ciclo.
    - Apoyo: desde foot strike inicial hasta toe-off estimado.
    - Oscilación: desde toe-off estimado hasta el siguiente foot strike.
    - Toe clearance: elevación máxima del toe durante la oscilación respecto al
      nivel de contacto estimado del ciclo, en pixeles.

Nota importante:
    La detección de toe-off es una estimación conservadora basada en la elevación
    vertical del toe/foot dentro de ciclos ya validados. El gráfico de control es
    obligatorio para revisar si las fases están bien cortadas.

Autor: pipeline preparado para análisis DLC de marcha murina.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PARAMETROS EDITABLES
# =============================================================================

FPS = 30.0

# Punto usado para estimar apoyo/oscilación y toe clearance.
# Recomendado inicialmente: "toe". Si el tracking del toe es malo, probar "foot".
BODY_PART_FOR_PHASE = "toe"

# En imágenes, el eje Y suele aumentar hacia abajo.
# Si el contacto con la cinta corresponde a valores altos de y, usar "max".
# Si el contacto corresponde a valores bajos de y, usar "min".
CONTACT_POLARITY = "max"

# Usar solo ciclos aceptados por el script 01.
ACCEPTED_ONLY = True

# Fracción mínima de datos finitos dentro de un ciclo para analizarlo.
MIN_FINITE_FRACTION = 0.70

# Para estimar el nivel de contacto:
#   polarity="max": mediana del porcentaje superior de valores y.
#   polarity="min": mediana del porcentaje inferior de valores y.
CONTACT_LEVEL_FRACTION = 0.20

# Umbral de elevación para definir toe-off.
# Se usa el máximo entre:
#   MIN_CLEARANCE_THRESHOLD_PX
#   TOE_OFF_THRESHOLD_FRACTION * rango_vertical_del_ciclo
TOE_OFF_THRESHOLD_FRACTION = 0.25
MIN_CLEARANCE_THRESHOLD_PX = 2.0

# Duraciones mínimas para aceptar apoyo/oscilación.
# A 30 FPS, 0.05 s ≈ 2 frames.
MIN_STANCE_DURATION_S = 0.05
MIN_SWING_DURATION_S = 0.05

# Número de frames consecutivos que deben superar el umbral para declarar toe-off.
SUSTAIN_FRAMES = 2

# Si True, muestra gráficos al terminar.
SHOW_PLOTS = False


# =============================================================================
# LECTURA Y VALIDACION
# =============================================================================

def read_clean_coords(clean_file: Path) -> pd.DataFrame:
    """Lee el archivo *_clean_coords.csv generado por el script 01."""
    clean_file = Path(clean_file)
    if not clean_file.exists():
        raise FileNotFoundError(f"No existe clean_coords: {clean_file}")

    df = pd.read_csv(clean_file)

    if "frame" in df.columns:
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
        if df["frame"].isna().any():
            raise ValueError("La columna frame contiene valores no numéricos.")
        df = df.set_index("frame", drop=True)
    else:
        df.index.name = "frame"

    df.index = df.index.astype(int)
    df = df.sort_index()
    return df


def infer_cycles_path(clean_file: Path) -> Path:
    """Infiere el nombre del archivo *_gait_cycles.csv desde *_clean_coords.csv."""
    clean_file = Path(clean_file)
    name = clean_file.name

    if name.endswith("_clean_coords.csv"):
        cycles_name = name.replace("_clean_coords.csv", "_gait_cycles.csv")
    else:
        cycles_name = clean_file.stem + "_gait_cycles.csv"

    return clean_file.with_name(cycles_name)


def read_gait_cycles(cycles_file: Path, accepted_only: bool = True) -> pd.DataFrame:
    """Lee el archivo *_gait_cycles.csv generado por el script 01."""
    cycles_file = Path(cycles_file)
    if not cycles_file.exists():
        raise FileNotFoundError(f"No existe gait_cycles: {cycles_file}")

    cycles = pd.read_csv(cycles_file)

    required = ["start_frame", "end_frame"]
    missing = [c for c in required if c not in cycles.columns]
    if missing:
        raise ValueError(
            "El archivo de ciclos no tiene las columnas requeridas: "
            + ", ".join(missing)
        )

    for col in ["cycle_id", "start_frame", "end_frame", "duration_frames", "duration_s", "accepted"]:
        if col in cycles.columns:
            cycles[col] = pd.to_numeric(cycles[col], errors="coerce")

    if "cycle_id" not in cycles.columns:
        cycles["cycle_id"] = np.arange(1, len(cycles) + 1, dtype=int)

    if accepted_only and "accepted" in cycles.columns:
        cycles = cycles[cycles["accepted"] == 1].copy()

    cycles = cycles.dropna(subset=["cycle_id", "start_frame", "end_frame"]).copy()
    cycles["cycle_id"] = cycles["cycle_id"].astype(int)
    cycles["start_frame"] = cycles["start_frame"].astype(int)
    cycles["end_frame"] = cycles["end_frame"].astype(int)
    cycles = cycles[cycles["end_frame"] > cycles["start_frame"]].copy()

    if "duration_frames" not in cycles.columns:
        cycles["duration_frames"] = cycles["end_frame"] - cycles["start_frame"]
    else:
        cycles["duration_frames"] = cycles["duration_frames"].fillna(
            cycles["end_frame"] - cycles["start_frame"]
        )

    cycles = cycles.sort_values(["start_frame", "end_frame"]).reset_index(drop=True)

    if cycles.empty:
        raise ValueError("No quedaron ciclos válidos para analizar.")

    return cycles


def validate_required_columns(clean: pd.DataFrame, bodypart: str) -> None:
    """Verifica que existan las coordenadas limpias del bodypart seleccionado."""
    missing = []
    for coord in ["x", "y"]:
        col = f"{bodypart}_{coord}"
        if col not in clean.columns:
            missing.append(col)

    if missing:
        raise ValueError(
            "Faltan columnas de coordenadas limpias en *_clean_coords.csv: "
            + ", ".join(missing)
        )


# =============================================================================
# CALCULO DE FASES Y TOE CLEARANCE
# =============================================================================

def finite_fraction(values: np.ndarray) -> float:
    """Fracción de valores finitos."""
    if len(values) == 0:
        return 0.0
    return float(np.isfinite(values).mean())


def estimate_contact_level(
    signal: np.ndarray,
    polarity: str,
    contact_fraction: float,
) -> float:
    """
    Estima el nivel vertical de contacto con la cinta dentro de un ciclo.

    Para polarity="max", el contacto corresponde a valores altos de y.
    Para polarity="min", el contacto corresponde a valores bajos de y.
    """
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return np.nan

    contact_fraction = float(np.clip(contact_fraction, 0.05, 0.50))
    n_contact = max(1, int(np.ceil(finite.size * contact_fraction)))
    sorted_values = np.sort(finite)

    if polarity == "max":
        contact_values = sorted_values[-n_contact:]
    elif polarity == "min":
        contact_values = sorted_values[:n_contact]
    else:
        raise ValueError("CONTACT_POLARITY debe ser 'max' o 'min'.")

    return float(np.nanmedian(contact_values))


def vertical_elevation_from_contact(
    signal: np.ndarray,
    contact_level: float,
    polarity: str,
) -> np.ndarray:
    """
    Convierte y en elevación positiva respecto al nivel de contacto.

    Para polarity="max": contacto alto en y, elevación = contacto - y.
    Para polarity="min": contacto bajo en y, elevación = y - contacto.
    """
    if polarity == "max":
        return contact_level - signal
    if polarity == "min":
        return signal - contact_level
    raise ValueError("CONTACT_POLARITY debe ser 'max' o 'min'.")


def first_sustained_crossing(
    elevation: np.ndarray,
    threshold: float,
    start_idx: int,
    end_idx_exclusive: int,
    sustain_frames: int,
) -> Optional[int]:
    """Devuelve el primer índice donde elevation supera el umbral de forma sostenida."""
    sustain_frames = max(1, int(sustain_frames))
    start_idx = max(0, int(start_idx))
    end_idx_exclusive = min(len(elevation), int(end_idx_exclusive))

    if end_idx_exclusive <= start_idx:
        return None

    last_start = end_idx_exclusive - sustain_frames + 1
    for idx in range(start_idx, max(start_idx, last_start)):
        window = elevation[idx:idx + sustain_frames]
        if len(window) < sustain_frames:
            continue
        if np.isfinite(window).all() and np.all(window >= threshold):
            return idx

    return None


def analyze_one_cycle(
    clean: pd.DataFrame,
    cycle: pd.Series,
    bodypart: str,
    fps: float,
    polarity: str,
    min_finite_fraction: float,
    contact_fraction: float,
    threshold_fraction: float,
    min_clearance_threshold_px: float,
    min_stance_s: float,
    min_swing_s: float,
    sustain_frames: int,
) -> Dict[str, object]:
    """Calcula variables temporales y toe clearance para un ciclo."""
    cycle_id = int(cycle["cycle_id"])
    start = int(cycle["start_frame"])
    end = int(cycle["end_frame"])
    duration_frames = int(end - start)
    stride_duration_s = duration_frames / float(fps)

    row: Dict[str, object] = {
        "cycle_id": cycle_id,
        "start_frame": start,
        "end_frame": end,
        "duration_frames": duration_frames,
        "stride_duration_s": stride_duration_s,
        "toe_off_frame": np.nan,
        "stance_duration_frames": np.nan,
        "swing_duration_frames": np.nan,
        "stance_duration_s": np.nan,
        "swing_duration_s": np.nan,
        "stance_percent": np.nan,
        "swing_percent": np.nan,
        "contact_level_px": np.nan,
        "toe_clearance_px": np.nan,
        "cycle_vertical_range_px": np.nan,
        "toe_off_threshold_px": np.nan,
        "finite_fraction_signal": np.nan,
        "accepted_temporal": 0,
        "reject_reason": "",
    }

    if duration_frames <= 0:
        row["reject_reason"] = "duracion_invalida"
        return row

    y_col = f"{bodypart}_y"
    available_frames = clean.index[(clean.index >= start) & (clean.index <= end)].to_numpy(dtype=int)
    if available_frames.size == 0:
        row["reject_reason"] = "sin_frames_en_clean_coords"
        return row

    signal = clean.loc[available_frames, y_col].to_numpy(dtype=float)
    frac = finite_fraction(signal)
    row["finite_fraction_signal"] = frac

    if frac < min_finite_fraction:
        row["reject_reason"] = "pocos_datos_finitos"
        return row

    finite = signal[np.isfinite(signal)]
    vertical_range = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else np.nan
    row["cycle_vertical_range_px"] = vertical_range

    if not np.isfinite(vertical_range) or vertical_range <= 0:
        row["reject_reason"] = "rango_vertical_invalido"
        return row

    contact_level = estimate_contact_level(
        signal=signal,
        polarity=polarity,
        contact_fraction=contact_fraction,
    )
    row["contact_level_px"] = contact_level

    if not np.isfinite(contact_level):
        row["reject_reason"] = "nivel_contacto_invalido"
        return row

    elevation = vertical_elevation_from_contact(
        signal=signal,
        contact_level=contact_level,
        polarity=polarity,
    )

    threshold = max(float(min_clearance_threshold_px), float(threshold_fraction) * vertical_range)
    row["toe_off_threshold_px"] = threshold

    min_stance_frames = max(1, int(round(float(min_stance_s) * float(fps))))
    min_swing_frames = max(1, int(round(float(min_swing_s) * float(fps))))

    # La búsqueda de toe-off se hace después de una fase inicial mínima de apoyo
    # y antes de dejar una fase mínima de oscilación.
    search_start_idx = min_stance_frames
    search_end_idx_exclusive = len(elevation) - min_swing_frames

    toe_off_idx = first_sustained_crossing(
        elevation=elevation,
        threshold=threshold,
        start_idx=search_start_idx,
        end_idx_exclusive=search_end_idx_exclusive,
        sustain_frames=sustain_frames,
    )

    if toe_off_idx is None:
        row["reject_reason"] = "toe_off_no_detectado"
        return row

    toe_off_frame = int(available_frames[toe_off_idx])
    stance_duration_frames = toe_off_frame - start
    swing_duration_frames = end - toe_off_frame

    if stance_duration_frames < min_stance_frames:
        row["reject_reason"] = "apoyo_demasiado_corto"
        return row

    if swing_duration_frames < min_swing_frames:
        row["reject_reason"] = "oscilacion_demasiado_corta"
        return row

    if duration_frames <= 0:
        row["reject_reason"] = "duracion_invalida"
        return row

    # Toe clearance se calcula SOLO durante la fase de oscilación estimada.
    swing_mask = available_frames >= toe_off_frame
    swing_signal = signal[swing_mask]
    swing_elevation = elevation[swing_mask]
    swing_elevation = swing_elevation[np.isfinite(swing_elevation)]

    if swing_elevation.size == 0:
        row["reject_reason"] = "sin_datos_en_oscilacion"
        return row

    toe_clearance = float(np.nanmax(swing_elevation))
    if not np.isfinite(toe_clearance):
        row["reject_reason"] = "toe_clearance_invalido"
        return row

    row["toe_off_frame"] = toe_off_frame
    row["stance_duration_frames"] = int(stance_duration_frames)
    row["swing_duration_frames"] = int(swing_duration_frames)
    row["stance_duration_s"] = float(stance_duration_frames / float(fps))
    row["swing_duration_s"] = float(swing_duration_frames / float(fps))
    row["stance_percent"] = float((stance_duration_frames / duration_frames) * 100.0)
    row["swing_percent"] = float((swing_duration_frames / duration_frames) * 100.0)
    row["toe_clearance_px"] = toe_clearance
    row["accepted_temporal"] = 1
    row["reject_reason"] = ""

    return row


def calculate_temporal_variables(
    clean: pd.DataFrame,
    cycles: pd.DataFrame,
    bodypart: str = BODY_PART_FOR_PHASE,
    fps: float = FPS,
    polarity: str = CONTACT_POLARITY,
    min_finite_fraction: float = MIN_FINITE_FRACTION,
    contact_fraction: float = CONTACT_LEVEL_FRACTION,
    threshold_fraction: float = TOE_OFF_THRESHOLD_FRACTION,
    min_clearance_threshold_px: float = MIN_CLEARANCE_THRESHOLD_PX,
    min_stance_s: float = MIN_STANCE_DURATION_S,
    min_swing_s: float = MIN_SWING_DURATION_S,
    sustain_frames: int = SUSTAIN_FRAMES,
) -> pd.DataFrame:
    """Calcula variables temporales y toe clearance para todos los ciclos."""
    rows = []
    for _, cyc in cycles.iterrows():
        rows.append(
            analyze_one_cycle(
                clean=clean,
                cycle=cyc,
                bodypart=bodypart,
                fps=fps,
                polarity=polarity,
                min_finite_fraction=min_finite_fraction,
                contact_fraction=contact_fraction,
                threshold_fraction=threshold_fraction,
                min_clearance_threshold_px=min_clearance_threshold_px,
                min_stance_s=min_stance_s,
                min_swing_s=min_swing_s,
                sustain_frames=sustain_frames,
            )
        )
    return pd.DataFrame(rows)


# =============================================================================
# RESUMEN POR VIDEO
# =============================================================================

def mean_sd_sem(values: pd.Series) -> Tuple[float, float, float, int]:
    """Calcula media, D.E., S.E.M. y n ignorando NaN."""
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(arr))

    if n == 0:
        return np.nan, np.nan, np.nan, 0
    if n == 1:
        return float(arr[0]), np.nan, np.nan, 1

    sd = float(np.std(arr, ddof=1))
    sem = float(sd / np.sqrt(n))
    return float(np.mean(arr)), sd, sem, n


def calculate_video_summary(by_cycle: pd.DataFrame) -> pd.DataFrame:
    """Resume variables por video usando solo ciclos temporales aceptados."""
    valid = by_cycle[by_cycle["accepted_temporal"] == 1].copy()

    row: Dict[str, object] = {
        "n_cycles_total": int(len(by_cycle)),
        "n_cycles_temporal_valid": int(len(valid)),
        "n_cycles_temporal_rejected": int(len(by_cycle) - len(valid)),
    }

    variables = [
        "stride_duration_s",
        "stance_duration_s",
        "swing_duration_s",
        "stance_percent",
        "swing_percent",
        "toe_clearance_px",
    ]

    for var in variables:
        mean, sd, sem, n = mean_sd_sem(valid[var] if var in valid.columns else pd.Series(dtype=float))
        row[f"{var}_n_valid"] = n
        row[f"{var}_mean"] = mean
        row[f"{var}_sd"] = sd
        row[f"{var}_sem"] = sem

    return pd.DataFrame([row])


# =============================================================================
# GRAFICOS DE CONTROL
# =============================================================================

def plot_temporal_control(
    clean: pd.DataFrame,
    by_cycle: pd.DataFrame,
    bodypart: str,
    output_png: Path,
    show: bool = False,
) -> None:
    """Grafica la señal vertical y las fases estimadas por ciclo."""
    y_col = f"{bodypart}_y"
    frames = clean.index.to_numpy(dtype=int)
    y = clean[y_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(frames, y, linewidth=1.0, label=f"{bodypart}_y")

    for _, row in by_cycle.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        accepted = int(row["accepted_temporal"]) == 1

        if accepted:
            toe_off = int(row["toe_off_frame"])
            ax.axvspan(start, toe_off, alpha=0.12, label="apoyo" if "apoyo" not in ax.get_legend_handles_labels()[1] else None)
            ax.axvspan(toe_off, end, alpha=0.06, label="oscilación" if "oscilación" not in ax.get_legend_handles_labels()[1] else None)
            ax.axvline(toe_off, linewidth=1.0, alpha=0.8, linestyle="--", label="toe-off" if "toe-off" not in ax.get_legend_handles_labels()[1] else None)
        else:
            ax.axvspan(start, end, alpha=0.04)

        ax.axvline(start, linewidth=0.6, alpha=0.5)
        ax.axvline(end, linewidth=0.6, alpha=0.5)

    ax.set_title(f"Control de fases temporales | punto={bodypart}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Coordenada y (px)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)

    if show:
        plt.show()
    plt.close(fig)


def plot_toe_clearance_control(
    by_cycle: pd.DataFrame,
    output_png: Path,
    show: bool = False,
) -> None:
    """Grafica toe clearance y porcentajes de apoyo/oscilación por ciclo."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    valid = by_cycle[by_cycle["accepted_temporal"] == 1].copy()
    rejected = by_cycle[by_cycle["accepted_temporal"] != 1].copy()

    if not valid.empty:
        axes[0].plot(valid["cycle_id"], valid["toe_clearance_px"], marker="o", linewidth=1.5)
        axes[1].plot(valid["cycle_id"], valid["stance_percent"], marker="o", linewidth=1.5)
        axes[2].plot(valid["cycle_id"], valid["swing_percent"], marker="o", linewidth=1.5)

    if not rejected.empty:
        for ax in axes:
            ax.scatter(rejected["cycle_id"], np.zeros(len(rejected)), marker="x", label="rechazado")

    axes[0].set_ylabel("Toe clearance (px)")
    axes[1].set_ylabel("Apoyo (%)")
    axes[2].set_ylabel("Oscilación (%)")
    axes[2].set_xlabel("Cycle ID")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.suptitle("Control por ciclo: toe clearance, apoyo y oscilación", y=0.995)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)

    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# EXPORTACION Y PIPELINE PRINCIPAL
# =============================================================================

def base_stem_from_clean_file(clean_file: Path) -> str:
    """Obtiene un stem base desde *_clean_coords.csv."""
    stem = clean_file.stem
    if stem.endswith("_clean_coords"):
        stem = stem[: -len("_clean_coords")]
    return stem.replace(" ", "_")


def write_params(
    params_file: Path,
    clean_file: Path,
    cycles_file: Path,
    fps: float,
    bodypart: str,
    polarity: str,
    accepted_only: bool,
    min_finite_fraction: float,
    contact_fraction: float,
    threshold_fraction: float,
    min_clearance_threshold_px: float,
    min_stance_s: float,
    min_swing_s: float,
    sustain_frames: int,
    n_cycles: int,
    n_valid: int,
) -> None:
    """Guarda parámetros de ejecución."""
    with open(params_file, "w", encoding="utf-8") as f:
        f.write("03_variables_temporales_y_toe_clearance.py\n")
        f.write(f"clean_file = {clean_file}\n")
        f.write(f"cycles_file = {cycles_file}\n")
        f.write(f"fps = {fps}\n")
        f.write(f"bodypart_for_phase = {bodypart}\n")
        f.write(f"contact_polarity = {polarity}\n")
        f.write(f"accepted_only = {accepted_only}\n")
        f.write(f"min_finite_fraction = {min_finite_fraction}\n")
        f.write(f"contact_level_fraction = {contact_fraction}\n")
        f.write(f"toe_off_threshold_fraction = {threshold_fraction}\n")
        f.write(f"min_clearance_threshold_px = {min_clearance_threshold_px}\n")
        f.write(f"min_stance_duration_s = {min_stance_s}\n")
        f.write(f"min_swing_duration_s = {min_swing_s}\n")
        f.write(f"sustain_frames = {sustain_frames}\n")
        f.write(f"n_cycles_analyzed = {n_cycles}\n")
        f.write(f"n_cycles_temporal_valid = {n_valid}\n")
        f.write("definitions = stride_duration: start_frame to end_frame; stance: start_frame to toe_off; swing: toe_off to end_frame; toe_clearance: max elevation during swing relative to contact level\n")


def run_pipeline(
    clean_file: Path,
    cycles_file: Optional[Path] = None,
    outdir: Optional[Path] = None,
    fps: float = FPS,
    bodypart: str = BODY_PART_FOR_PHASE,
    polarity: str = CONTACT_POLARITY,
    accepted_only: bool = ACCEPTED_ONLY,
    min_finite_fraction: float = MIN_FINITE_FRACTION,
    contact_fraction: float = CONTACT_LEVEL_FRACTION,
    threshold_fraction: float = TOE_OFF_THRESHOLD_FRACTION,
    min_clearance_threshold_px: float = MIN_CLEARANCE_THRESHOLD_PX,
    min_stance_s: float = MIN_STANCE_DURATION_S,
    min_swing_s: float = MIN_SWING_DURATION_S,
    sustain_frames: int = SUSTAIN_FRAMES,
    show_plots: bool = SHOW_PLOTS,
) -> Dict[str, Path]:
    """Ejecuta la tercera parte completa del pipeline."""
    clean_file = Path(clean_file)

    if cycles_file is None:
        cycles_file = infer_cycles_path(clean_file)
    cycles_file = Path(cycles_file)

    if outdir is None:
        outdir = Path("salida_03_temporal")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = base_stem_from_clean_file(clean_file)

    print("\n=== 03 VARIABLES TEMPORALES Y TOE CLEARANCE ===")
    print(f"Coordenadas limpias: {clean_file}")
    print(f"Ciclos de marcha:    {cycles_file}")
    print(f"Carpeta salida:      {outdir}")
    print(f"FPS:                 {fps}")
    print(f"Punto usado:         {bodypart}")
    print(f"Polaridad contacto:  {polarity}")
    print(f"Solo aceptados:      {accepted_only}")

    clean = read_clean_coords(clean_file)
    validate_required_columns(clean, bodypart)
    cycles = read_gait_cycles(cycles_file, accepted_only=accepted_only)

    by_cycle = calculate_temporal_variables(
        clean=clean,
        cycles=cycles,
        bodypart=bodypart,
        fps=fps,
        polarity=polarity,
        min_finite_fraction=min_finite_fraction,
        contact_fraction=contact_fraction,
        threshold_fraction=threshold_fraction,
        min_clearance_threshold_px=min_clearance_threshold_px,
        min_stance_s=min_stance_s,
        min_swing_s=min_swing_s,
        sustain_frames=sustain_frames,
    )

    summary = calculate_video_summary(by_cycle)

    by_cycle_csv = outdir / f"{stem}_gait_temporal_by_cycle.csv"
    summary_csv = outdir / f"{stem}_gait_temporal_video_summary.csv"
    temporal_png = outdir / f"{stem}_gait_temporal_control.png"
    toe_clearance_png = outdir / f"{stem}_toe_clearance_control.png"
    params_txt = outdir / f"{stem}_temporal_params.txt"

    by_cycle.to_csv(by_cycle_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    plot_temporal_control(
        clean=clean,
        by_cycle=by_cycle,
        bodypart=bodypart,
        output_png=temporal_png,
        show=show_plots,
    )

    plot_toe_clearance_control(
        by_cycle=by_cycle,
        output_png=toe_clearance_png,
        show=show_plots,
    )

    n_valid = int((by_cycle["accepted_temporal"] == 1).sum())
    write_params(
        params_file=params_txt,
        clean_file=clean_file,
        cycles_file=cycles_file,
        fps=fps,
        bodypart=bodypart,
        polarity=polarity,
        accepted_only=accepted_only,
        min_finite_fraction=min_finite_fraction,
        contact_fraction=contact_fraction,
        threshold_fraction=threshold_fraction,
        min_clearance_threshold_px=min_clearance_threshold_px,
        min_stance_s=min_stance_s,
        min_swing_s=min_swing_s,
        sustain_frames=sustain_frames,
        n_cycles=len(by_cycle),
        n_valid=n_valid,
    )

    print("\nListo.")
    print(f"  Ciclos analizados:          {len(by_cycle)}")
    print(f"  Ciclos temporales válidos:  {n_valid}")
    print("\nArchivos generados:")
    print(f"  {by_cycle_csv}")
    print(f"  {summary_csv}")
    print(f"  {temporal_png}")
    print(f"  {toe_clearance_png}")
    print(f"  {params_txt}")

    return {
        "gait_temporal_by_cycle": by_cycle_csv,
        "gait_temporal_video_summary": summary_csv,
        "gait_temporal_control": temporal_png,
        "toe_clearance_control": toe_clearance_png,
        "params": params_txt,
    }


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="03 - Calcula tiempo de zancada, apoyo %, oscilación % y toe clearance desde ciclos validados."
    )

    parser.add_argument(
        "clean_coords",
        type=str,
        help="Archivo *_clean_coords.csv generado por 01_preprocesamiento_y_ciclos.py",
    )
    parser.add_argument(
        "--cycles",
        type=str,
        default=None,
        help="Archivo *_gait_cycles.csv. Si se omite, se infiere desde clean_coords.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="salida_03_temporal",
        help="Carpeta de salida.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=FPS,
        help="Frames por segundo del video. Por defecto: 30.",
    )
    parser.add_argument(
        "--bodypart",
        type=str,
        default=BODY_PART_FOR_PHASE,
        choices=["toe", "foot"],
        help="Punto usado para apoyo/oscilación y toe clearance. Por defecto: toe.",
    )
    parser.add_argument(
        "--polarity",
        type=str,
        default=CONTACT_POLARITY,
        choices=["max", "min"],
        help="Polaridad del contacto con la cinta. max si contacto = y alto; min si contacto = y bajo.",
    )
    parser.add_argument(
        "--all-cycles",
        action="store_true",
        help="Usar todos los ciclos del archivo, incluidos rechazados por el script 01. No recomendado.",
    )
    parser.add_argument(
        "--min-finite-fraction",
        type=float,
        default=MIN_FINITE_FRACTION,
        help="Fracción mínima de datos finitos por ciclo. Por defecto: 0.70.",
    )
    parser.add_argument(
        "--contact-fraction",
        type=float,
        default=CONTACT_LEVEL_FRACTION,
        help="Fracción de valores extremos usada para estimar nivel de contacto. Por defecto: 0.20.",
    )
    parser.add_argument(
        "--threshold-fraction",
        type=float,
        default=TOE_OFF_THRESHOLD_FRACTION,
        help="Fracción del rango vertical usada como umbral de toe-off. Por defecto: 0.25.",
    )
    parser.add_argument(
        "--min-clearance-threshold",
        type=float,
        default=MIN_CLEARANCE_THRESHOLD_PX,
        help="Umbral mínimo absoluto en pixeles para toe-off. Por defecto: 2.0.",
    )
    parser.add_argument(
        "--min-stance-s",
        type=float,
        default=MIN_STANCE_DURATION_S,
        help="Duración mínima de apoyo para aceptar un ciclo. Por defecto: 0.05 s.",
    )
    parser.add_argument(
        "--min-swing-s",
        type=float,
        default=MIN_SWING_DURATION_S,
        help="Duración mínima de oscilación para aceptar un ciclo. Por defecto: 0.05 s.",
    )
    parser.add_argument(
        "--sustain-frames",
        type=int,
        default=SUSTAIN_FRAMES,
        help="Frames consecutivos sobre umbral para detectar toe-off. Por defecto: 2.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Mostrar gráficos además de guardarlos.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    try:
        run_pipeline(
            clean_file=Path(args.clean_coords),
            cycles_file=Path(args.cycles) if args.cycles else None,
            outdir=Path(args.outdir),
            fps=float(args.fps),
            bodypart=args.bodypart,
            polarity=args.polarity,
            accepted_only=not bool(args.all_cycles),
            min_finite_fraction=float(args.min_finite_fraction),
            contact_fraction=float(args.contact_fraction),
            threshold_fraction=float(args.threshold_fraction),
            min_clearance_threshold_px=float(args.min_clearance_threshold),
            min_stance_s=float(args.min_stance_s),
            min_swing_s=float(args.min_swing_s),
            sustain_frames=int(args.sustain_frames),
            show_plots=bool(args.show_plots),
        )
        return 0
    except Exception as exc:
        print("\nERROR:", str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
