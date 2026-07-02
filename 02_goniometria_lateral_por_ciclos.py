#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_goniometria_lateral_por_ciclos.py

Segunda parte del pipeline para análisis de marcha murina con DeepLabCut.
Este script SOLO hace goniometría lateral usando las salidas del script 01.

Entrada esperada:
    *_clean_coords.csv
    *_gait_cycles.csv

Este script hace:
    1. Leer coordenadas limpias del script 01.
    2. Leer ciclos de marcha detectados/validados del script 01.
    3. Calcular ángulos frame a frame:
        - hip
        - knee
        - ankle
        - foot
    4. Exportar ángulos frame a frame.
    5. Recortar perfiles angulares por ciclo.
    6. Normalizar perfiles angulares a 0-100% del ciclo.
    7. Exportar perfiles angulares por ciclo.
    8. Calcular rangos angulares por ciclo: max - min.
    9. Exportar rangos angulares por ciclo.
   10. Exportar resumen por video de rangos: media, D.E. y S.E.M.
   11. Graficar perfiles angulares.
   12. Graficar control de ciclos sobre señales angulares.

Este script NO detecta ciclos nuevos, NO calcula stance/swing, NO calcula toe clearance
final y NO hace estadística grupal.

Uso recomendado:
    python 02_goniometria_lateral_por_ciclos.py "salida_01/archivo_clean_coords.csv" \
        --cycles "salida_01/archivo_gait_cycles.csv" --outdir salida_02_angulos

Si el archivo de ciclos está en la misma carpeta y tiene el nombre estándar del script 01,
se puede omitir --cycles:
    python 02_goniometria_lateral_por_ciclos.py "salida_01/archivo_clean_coords.csv"

Definición angular por defecto: internal
    hip   = ángulo crest-hip-knee
    knee  = ángulo hip-knee-ankle
    ankle = ángulo knee-ankle-foot
    foot  = ángulo ankle-foot-toe

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

BODY_PARTS = ["crest", "hip", "knee", "ankle", "foot", "toe"]
ANGLE_NAMES = ["hip", "knee", "ankle", "foot"]
ANGLE_COLS = [f"{a}_angle_deg" for a in ANGLE_NAMES]

# 101 puntos = 0, 1, 2, ..., 100% del ciclo.
NORM_POINTS = 101

# Definición angular.
# "internal": ángulos articulares internos usando 3 puntos.
ANGLE_DEFINITION = "internal"

# Usar solo ciclos aceptados del script 01.
ACCEPTED_ONLY = True

# Rechazar rangos si la fracción de datos finitos de ese ángulo es menor que esto.
# El perfil se exporta igual, pero el rango queda como NaN si no cumple.
MIN_FINITE_FRACTION_FOR_RANGE = 0.70

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

    # El script 01 guarda el índice como columna frame.
    if "frame" in df.columns:
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
        if df["frame"].isna().any():
            raise ValueError("La columna frame contiene valores no numéricos.")
        df = df.set_index("frame", drop=True)
    else:
        # Respaldo: si no existe frame, usar el índice del CSV.
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

    # Convertir columnas críticas.
    for col in ["start_frame", "end_frame", "duration_frames"]:
        if col in cycles.columns:
            cycles[col] = pd.to_numeric(cycles[col], errors="coerce")

    if "cycle_id" not in cycles.columns:
        cycles["cycle_id"] = np.arange(1, len(cycles) + 1, dtype=int)
    else:
        cycles["cycle_id"] = pd.to_numeric(cycles["cycle_id"], errors="coerce")

    if accepted_only and "accepted" in cycles.columns:
        cycles = cycles[cycles["accepted"] == 1].copy()

    cycles = cycles.dropna(subset=["cycle_id", "start_frame", "end_frame"]).copy()

    cycles["cycle_id"] = cycles["cycle_id"].astype(int)
    cycles["start_frame"] = cycles["start_frame"].astype(int)
    cycles["end_frame"] = cycles["end_frame"].astype(int)

    cycles = cycles[cycles["end_frame"] > cycles["start_frame"]].copy()
    cycles = cycles.sort_values(["start_frame", "end_frame"]).reset_index(drop=True)

    if cycles.empty:
        raise ValueError("No quedaron ciclos válidos para analizar.")

    return cycles


def validate_required_columns(clean: pd.DataFrame, bodyparts: List[str]) -> None:
    """Verifica que existan las coordenadas limpias bp_x y bp_y."""
    missing = []
    for bp in bodyparts:
        for coord in ["x", "y"]:
            col = f"{bp}_{coord}"
            if col not in clean.columns:
                missing.append(col)

    if missing:
        raise ValueError(
            "Faltan columnas de coordenadas limpias en *_clean_coords.csv: "
            + ", ".join(missing)
        )


# =============================================================================
# CALCULO DE ANGULOS
# =============================================================================

def _xy(clean: pd.DataFrame, bp: str) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve x,y de un punto anatómico como arrays float."""
    x = clean[f"{bp}_x"].to_numpy(dtype=float)
    y = clean[f"{bp}_y"].to_numpy(dtype=float)
    return x, y


def internal_angle_deg(clean: pd.DataFrame, a: str, b: str, c: str) -> np.ndarray:
    """
    Calcula el ángulo interno ABC en grados.
    El vértice es b.
    Ejemplo: knee = hip-knee-ankle.
    """
    ax, ay = _xy(clean, a)
    bx, by = _xy(clean, b)
    cx, cy = _xy(clean, c)

    v1x = ax - bx
    v1y = ay - by
    v2x = cx - bx
    v2y = cy - by

    dot = v1x * v2x + v1y * v2y
    n1 = np.sqrt(v1x**2 + v1y**2)
    n2 = np.sqrt(v2x**2 + v2y**2)
    denom = n1 * n2

    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = dot / denom

    cosang = np.clip(cosang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))

    invalid = (~np.isfinite(denom)) | (denom <= 0)
    angle[invalid] = np.nan

    return angle


def orientation_angle_deg(clean: pd.DataFrame, proximal: str, distal: str, invert_y: bool = True) -> np.ndarray:
    """
    Calcula orientación de un segmento respecto al eje horizontal.
    Se incluye como función auxiliar para futuras variantes, pero el modo por defecto
    del script usa ángulos internos.
    """
    px, py = _xy(clean, proximal)
    dx, dy = _xy(clean, distal)

    vx = dx - px
    vy = dy - py
    if invert_y:
        vy = -vy

    angle = np.degrees(np.arctan2(vy, vx))
    angle[~np.isfinite(vx) | ~np.isfinite(vy)] = np.nan
    return angle


def calculate_frame_angles(clean: pd.DataFrame, angle_definition: str = ANGLE_DEFINITION) -> pd.DataFrame:
    """Calcula ángulos frame a frame."""
    if angle_definition != "internal":
        raise ValueError("Por ahora angle_definition soportado: 'internal'.")

    angles = pd.DataFrame(index=clean.index.copy())
    angles.index.name = "frame"

    if "time_s" in clean.columns:
        angles["time_s"] = pd.to_numeric(clean["time_s"], errors="coerce")

    # Definiciones por defecto.
    angles["hip_angle_deg"] = internal_angle_deg(clean, "crest", "hip", "knee")
    angles["knee_angle_deg"] = internal_angle_deg(clean, "hip", "knee", "ankle")
    angles["ankle_angle_deg"] = internal_angle_deg(clean, "knee", "ankle", "foot")
    angles["foot_angle_deg"] = internal_angle_deg(clean, "ankle", "foot", "toe")

    return angles


# =============================================================================
# NORMALIZACION DE PERFILES ANGULARES
# =============================================================================

def normalize_one_signal(frames: np.ndarray, values: np.ndarray, norm_percent: np.ndarray) -> np.ndarray:
    """Normaliza una señal de un ciclo a 0-100% usando interpolación lineal."""
    frames = np.asarray(frames, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(frames) < 2 or len(values) < 2:
        return np.full_like(norm_percent, np.nan, dtype=float)

    if frames[-1] == frames[0]:
        return np.full_like(norm_percent, np.nan, dtype=float)

    cycle_percent = (frames - frames[0]) / (frames[-1] - frames[0]) * 100.0
    finite = np.isfinite(values)

    if finite.sum() < 2:
        return np.full_like(norm_percent, np.nan, dtype=float)

    return np.interp(norm_percent, cycle_percent[finite], values[finite])


def normalize_angle_profiles(
    angles: pd.DataFrame,
    cycles: pd.DataFrame,
    norm_points: int = NORM_POINTS,
) -> pd.DataFrame:
    """Recorta cada ciclo y normaliza los perfiles angulares a 0-100%."""
    norm_percent = np.linspace(0, 100, norm_points)
    rows = []

    available_frames = set(angles.index.astype(int).tolist())

    for _, cyc in cycles.iterrows():
        cycle_id = int(cyc["cycle_id"])
        start = int(cyc["start_frame"])
        end = int(cyc["end_frame"])

        cycle_frames = np.arange(start, end + 1, dtype=int)
        cycle_frames = np.array([f for f in cycle_frames if f in available_frames], dtype=int)

        if len(cycle_frames) < 2:
            continue

        temp = pd.DataFrame({"percent_gait_cycle": norm_percent})
        temp["cycle_id"] = cycle_id
        temp["start_frame"] = start
        temp["end_frame"] = end

        if "duration_frames" in cyc.index and pd.notna(cyc["duration_frames"]):
            temp["duration_frames"] = int(cyc["duration_frames"])
        else:
            temp["duration_frames"] = end - start

        if "duration_s" in cyc.index and pd.notna(cyc["duration_s"]):
            temp["duration_s"] = float(cyc["duration_s"])

        for angle_col in ANGLE_COLS:
            values = angles.loc[cycle_frames, angle_col].to_numpy(dtype=float)
            temp[angle_col] = normalize_one_signal(cycle_frames, values, norm_percent)

        first_cols = ["cycle_id", "percent_gait_cycle", "start_frame", "end_frame", "duration_frames"]
        if "duration_s" in temp.columns:
            first_cols.append("duration_s")
        other_cols = [c for c in temp.columns if c not in first_cols]
        temp = temp[first_cols + other_cols]

        rows.append(temp)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# RANGOS ANGULARES POR CICLO
# =============================================================================

def calculate_cycle_angle_ranges(
    angles: pd.DataFrame,
    cycles: pd.DataFrame,
    min_finite_fraction: float = MIN_FINITE_FRACTION_FOR_RANGE,
) -> pd.DataFrame:
    """Calcula min, max, rango, media y SD angular por ciclo."""
    rows = []
    available_frames = set(angles.index.astype(int).tolist())

    for _, cyc in cycles.iterrows():
        cycle_id = int(cyc["cycle_id"])
        start = int(cyc["start_frame"])
        end = int(cyc["end_frame"])
        cycle_frames = np.arange(start, end + 1, dtype=int)
        cycle_frames = np.array([f for f in cycle_frames if f in available_frames], dtype=int)

        row: Dict[str, object] = {
            "cycle_id": cycle_id,
            "start_frame": start,
            "end_frame": end,
            "duration_frames": int(cyc["duration_frames"]) if "duration_frames" in cyc.index and pd.notna(cyc["duration_frames"]) else end - start,
        }

        if "duration_s" in cyc.index and pd.notna(cyc["duration_s"]):
            row["duration_s"] = float(cyc["duration_s"])

        for angle_name, angle_col in zip(ANGLE_NAMES, ANGLE_COLS):
            values = angles.loc[cycle_frames, angle_col].to_numpy(dtype=float) if len(cycle_frames) else np.array([], dtype=float)
            finite = np.isfinite(values)
            finite_fraction = float(finite.mean()) if len(values) else 0.0

            row[f"{angle_name}_finite_fraction"] = finite_fraction

            if len(values) == 0 or finite.sum() < 2 or finite_fraction < min_finite_fraction:
                row[f"{angle_name}_min_deg"] = np.nan
                row[f"{angle_name}_max_deg"] = np.nan
                row[f"{angle_name}_range_deg"] = np.nan
                row[f"{angle_name}_mean_deg"] = np.nan
                row[f"{angle_name}_sd_deg"] = np.nan
                continue

            finite_values = values[finite]
            row[f"{angle_name}_min_deg"] = float(np.nanmin(finite_values))
            row[f"{angle_name}_max_deg"] = float(np.nanmax(finite_values))
            row[f"{angle_name}_range_deg"] = float(np.nanmax(finite_values) - np.nanmin(finite_values))
            row[f"{angle_name}_mean_deg"] = float(np.nanmean(finite_values))
            row[f"{angle_name}_sd_deg"] = float(np.nanstd(finite_values, ddof=1)) if finite.sum() >= 2 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_angle_range_summary(ranges: pd.DataFrame) -> pd.DataFrame:
    """
    Resume los rangos angulares por video.

    Importante:
    - La unidad estadística aquí son los ciclos válidos dentro del video.
    - Para análisis grupal posterior, usar una fila por video/animal/estadio,
      no cada ciclo como observación independiente.
    - El S.E.M. se calcula como D.E. / sqrt(n).
    """
    row: Dict[str, object] = {
        "n_cycles_total": int(len(ranges)),
    }

    for angle_name in ANGLE_NAMES:
        col = f"{angle_name}_range_deg"
        if col not in ranges.columns:
            row[f"{angle_name}_range_n_valid"] = 0
            row[f"{angle_name}_range_mean_deg"] = np.nan
            row[f"{angle_name}_range_sd_deg"] = np.nan
            row[f"{angle_name}_range_sem_deg"] = np.nan
            continue

        values = pd.to_numeric(ranges[col], errors="coerce").dropna().to_numpy(dtype=float)
        n = int(len(values))
        row[f"{angle_name}_range_n_valid"] = n

        if n == 0:
            row[f"{angle_name}_range_mean_deg"] = np.nan
            row[f"{angle_name}_range_sd_deg"] = np.nan
            row[f"{angle_name}_range_sem_deg"] = np.nan
        elif n == 1:
            row[f"{angle_name}_range_mean_deg"] = float(values[0])
            row[f"{angle_name}_range_sd_deg"] = np.nan
            row[f"{angle_name}_range_sem_deg"] = np.nan
        else:
            sd = float(np.std(values, ddof=1))
            row[f"{angle_name}_range_mean_deg"] = float(np.mean(values))
            row[f"{angle_name}_range_sd_deg"] = sd
            row[f"{angle_name}_range_sem_deg"] = float(sd / np.sqrt(n))

    return pd.DataFrame([row])


# =============================================================================
# GRAFICOS
# =============================================================================

def plot_angle_profiles(profiles: pd.DataFrame, output_png: Path, show: bool = False) -> None:
    """Grafica perfiles angulares normalizados por ciclo y promedio."""
    if profiles.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, angle_name, angle_col in zip(axes, ANGLE_NAMES, ANGLE_COLS):
        for cycle_id, sub in profiles.groupby("cycle_id"):
            ax.plot(
                sub["percent_gait_cycle"],
                sub[angle_col],
                linewidth=0.8,
                alpha=0.25,
            )

        mean_profile = profiles.groupby("percent_gait_cycle", as_index=False)[angle_col].mean()
        ax.plot(
            mean_profile["percent_gait_cycle"],
            mean_profile[angle_col],
            linewidth=2.5,
            label="promedio",
        )

        ax.set_title(f"{angle_name} angle")
        ax.set_xlabel("Ciclo de marcha (%)")
        ax.set_ylabel("Ángulo (°)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("Perfiles angulares normalizados por ciclo", y=0.995)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)

    if show:
        plt.show()
    plt.close(fig)


def plot_angle_cycle_control(
    angles: pd.DataFrame,
    cycles: pd.DataFrame,
    output_png: Path,
    show: bool = False,
) -> None:
    """Grafica control de ciclos sobre las señales angulares frame a frame."""
    frames = angles.index.to_numpy(dtype=int)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    for ax, angle_name, angle_col in zip(axes, ANGLE_NAMES, ANGLE_COLS):
        ax.plot(frames, angles[angle_col], linewidth=1.0, label=angle_name)

        for _, cyc in cycles.iterrows():
            start = int(cyc["start_frame"])
            end = int(cyc["end_frame"])
            ax.axvspan(start, end, alpha=0.08)
            ax.axvline(start, linewidth=0.5, alpha=0.35)

        ax.set_ylabel("Ángulo (°)")
        ax.set_title(f"Control de ciclos sobre {angle_name}")
        ax.grid(True, alpha=0.20)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"Control de ciclos usados para goniometría | ciclos={len(cycles)}", y=0.995)
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
    angle_definition: str,
    norm_points: int,
    accepted_only: bool,
    min_finite_fraction: float,
    n_cycles: int,
) -> None:
    """Guarda parámetros de ejecución."""
    with open(params_file, "w", encoding="utf-8") as f:
        f.write("02_goniometria_lateral_por_ciclos.py\n")
        f.write(f"clean_file = {clean_file}\n")
        f.write(f"cycles_file = {cycles_file}\n")
        f.write(f"angle_definition = {angle_definition}\n")
        f.write("angle_definitions_internal = hip crest-hip-knee; knee hip-knee-ankle; ankle knee-ankle-foot; foot ankle-foot-toe\n")
        f.write(f"norm_points = {norm_points}\n")
        f.write(f"accepted_only = {accepted_only}\n")
        f.write(f"min_finite_fraction_for_range = {min_finite_fraction}\n")
        f.write(f"n_cycles_analyzed = {n_cycles}\n")


def run_pipeline(
    clean_file: Path,
    cycles_file: Optional[Path] = None,
    outdir: Optional[Path] = None,
    angle_definition: str = ANGLE_DEFINITION,
    norm_points: int = NORM_POINTS,
    accepted_only: bool = ACCEPTED_ONLY,
    min_finite_fraction: float = MIN_FINITE_FRACTION_FOR_RANGE,
    show_plots: bool = SHOW_PLOTS,
) -> Dict[str, Path]:
    """Ejecuta la segunda parte completa del pipeline."""
    clean_file = Path(clean_file)

    if cycles_file is None:
        cycles_file = infer_cycles_path(clean_file)
    cycles_file = Path(cycles_file)

    if outdir is None:
        outdir = Path("salida_02_angulos")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = base_stem_from_clean_file(clean_file)

    print("\n=== 02 GONIOMETRIA LATERAL POR CICLOS ===")
    print(f"Coordenadas limpias: {clean_file}")
    print(f"Ciclos de marcha:    {cycles_file}")
    print(f"Carpeta salida:      {outdir}")
    print(f"Definición angular:  {angle_definition}")
    print(f"Normalización:       {norm_points} puntos")
    print(f"Solo aceptados:      {accepted_only}")

    clean = read_clean_coords(clean_file)
    validate_required_columns(clean, BODY_PARTS)

    cycles = read_gait_cycles(cycles_file, accepted_only=accepted_only)
    angles = calculate_frame_angles(clean, angle_definition=angle_definition)

    profiles = normalize_angle_profiles(
        angles=angles,
        cycles=cycles,
        norm_points=norm_points,
    )

    ranges = calculate_cycle_angle_ranges(
        angles=angles,
        cycles=cycles,
        min_finite_fraction=min_finite_fraction,
    )

    if profiles.empty:
        raise ValueError("No se pudieron construir perfiles angulares normalizados. Revisar ciclos y coordenadas.")

    frame_angles_csv = outdir / f"{stem}_frame_angles.csv"
    profiles_csv = outdir / f"{stem}_cycle_angle_profiles.csv"
    ranges_csv = outdir / f"{stem}_cycle_angle_ranges.csv"
    range_summary_csv = outdir / f"{stem}_angle_range_summary.csv"
    profiles_png = outdir / f"{stem}_angle_profiles.png"
    control_png = outdir / f"{stem}_angle_cycle_control.png"
    params_txt = outdir / f"{stem}_goniometry_params.txt"

    angles.to_csv(frame_angles_csv, index=True)
    profiles.to_csv(profiles_csv, index=False)
    range_summary = calculate_angle_range_summary(ranges)

    ranges.to_csv(ranges_csv, index=False)
    range_summary.to_csv(range_summary_csv, index=False)

    plot_angle_profiles(profiles, profiles_png, show=show_plots)
    plot_angle_cycle_control(angles, cycles, control_png, show=show_plots)

    write_params(
        params_file=params_txt,
        clean_file=clean_file,
        cycles_file=cycles_file,
        angle_definition=angle_definition,
        norm_points=norm_points,
        accepted_only=accepted_only,
        min_finite_fraction=min_finite_fraction,
        n_cycles=len(cycles),
    )

    print("\nResumen:")
    print(f"  Frames analizados:              {len(angles)}")
    print(f"  Ciclos angulares analizados:    {len(cycles)}")
    print(f"  Filas perfiles normalizados:    {len(profiles)}")
    print(f"  Filas rangos por ciclo:         {len(ranges)}")
    print("  Resumen de rangos:              media, D.E. y S.E.M. por video")

    print("\nArchivos generados:")
    print(f"  {frame_angles_csv}")
    print(f"  {profiles_csv}")
    print(f"  {ranges_csv}")
    print(f"  {range_summary_csv}")
    print(f"  {profiles_png}")
    print(f"  {control_png}")
    print(f"  {params_txt}")
    print("\nIMPORTANTE: revisar el PNG de control antes de usar los rangos angulares.\n")

    return {
        "frame_angles": frame_angles_csv,
        "cycle_angle_profiles": profiles_csv,
        "cycle_angle_ranges": ranges_csv,
        "angle_range_summary": range_summary_csv,
        "angle_profiles_png": profiles_png,
        "angle_cycle_control_png": control_png,
        "params": params_txt,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segunda parte: calcular goniometría lateral por ciclos ya detectados."
    )
    parser.add_argument(
        "clean_file",
        type=str,
        help="Archivo *_clean_coords.csv generado por 01_preprocesamiento_y_ciclos.py.",
    )
    parser.add_argument(
        "--cycles",
        type=str,
        default=None,
        help="Archivo *_gait_cycles.csv. Si se omite, se infiere desde clean_file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="salida_02_angulos",
        help="Carpeta de salida. Default: salida_02_angulos.",
    )
    parser.add_argument(
        "--angle-definition",
        type=str,
        default=ANGLE_DEFINITION,
        choices=["internal"],
        help="Definición angular. Default: internal.",
    )
    parser.add_argument(
        "--norm-points",
        type=int,
        default=NORM_POINTS,
        help="Puntos para normalizar cada ciclo. Default: 101.",
    )
    parser.add_argument(
        "--all-cycles",
        action="store_true",
        help="Usar todos los ciclos del archivo, incluidos rechazados. No recomendado.",
    )
    parser.add_argument(
        "--min-finite-fraction",
        type=float,
        default=MIN_FINITE_FRACTION_FOR_RANGE,
        help="Fracción mínima de datos finitos para calcular rango angular. Default: 0.70.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Mostrar gráficos al terminar.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    try:
        run_pipeline(
            clean_file=Path(args.clean_file),
            cycles_file=Path(args.cycles) if args.cycles else None,
            outdir=Path(args.outdir),
            angle_definition=args.angle_definition,
            norm_points=args.norm_points,
            accepted_only=not args.all_cycles,
            min_finite_fraction=args.min_finite_fraction,
            show_plots=args.show_plots,
        )
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
