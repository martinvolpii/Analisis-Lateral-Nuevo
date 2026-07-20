#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_preprocesamiento_y_ciclos.py

Detector multiseñal de ciclos de marcha para el proyecto P30 (DeepLabCut, vista lateral).

Versión validada visualmente en los 10 videos P30 el 2026-07-20.

Principio de detección
---------------------
1. Se filtran coordenadas por likelihood >= 0.70 y se interpolan solo gaps cortos.
2. La señal primaria de evento es la posición horizontal del toe relativa a la cadera:
       toe_x - hip_x
3. Cada mínimo candidato del Toe se acepta por una de dos vías:
   A) confirmación fuerte por un mínimo cercano del ángulo de rodilla hip-knee-ankle;
   B) confirmación alternativa conjunta por un extremo de knee_x-hip_x Y un extremo
      del ángulo de cadera crest-hip-knee.
4. No se impone un número mínimo de ciclos y nunca se insertan ciclos sintéticos.
5. Los ciclos se construyen exclusivamente entre eventos consecutivos detectados.
6. Los intervalos anómalos se marcan para auditoría, no se corrigen automáticamente.

Decisiones metodológicas fijadas para P30
-----------------------------------------
- FPS experimental: 60 Hz.
- LIKELIHOOD_MIN: 0.70.
- Suavizado para detección: 5 frames.
- Suavizado de coordenadas exportadas para goniometría/temporal: 5 frames.
  La versión previa usaba 11 frames; se redujo porque varios ciclos reales duran
  solo 5-8 frames y una ventana de 11 frames atenuaba fuertemente la cinemática.
- Duración mínima de ciclo: 0.08 s (5 frames a 60 Hz), como límite de resolución,
  no como objetivo de cadencia.
- Duración máxima de ciclo: 1.50 s.
- Normalización opcional del ciclo: 101 puntos (0-100%).

Salidas compatibles con scripts 02 y 03
----------------------------------------
*_clean_coords.csv
*_events_detected.csv
*_gait_cycles.csv
*_normalized_cycles.csv
*_cycle_detection_signals.csv
*_rejected_event_candidates.csv
*_cycle_detection_check.png
*_params.txt

Uso:
    python 01_preprocesamiento_y_ciclos.py archivo_DLC.h5 --outdir salida_01

Este script mantiene el esquema de columnas requerido por los scripts 02 y 03.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, savgol_filter
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Este script requiere scipy (scipy.signal.find_peaks/savgol_filter).") from exc

ALGORITHM_VERSION = "P30_multisignal_validated_v3_2026-07-20"

FPS = 60.0
BODY_PARTS = ["crest", "hip", "knee", "ankle", "foot", "toe"]
COORDS = ["x", "y", "likelihood"]

LIKELIHOOD_MIN = 0.70
MAX_GAP_INTERPOLATION = 10
ANALYSIS_SMOOTH_WINDOW = 5
DETECTION_SMOOTH_WINDOW = 5

MIN_CYCLE_DURATION_S = 0.08
MAX_CYCLE_DURATION_S = 1.50
NORM_POINTS = 101
DEFAULT_PROMINENCE_Z = 0.10

# Metadatos del experimento P30 Day 1.
# No modifican el algoritmo; quedan registrados para trazabilidad.
DEFAULT_TREADMILL_SPEED_CM_S = 20.0
DEFAULT_SEX = "male"
DEFAULT_LITTER_ID = "P30_single_shared_litter"

def sanitize_stem(path: Path) -> str:
    """Genera un stem seguro para nombres de salida."""
    return path.stem.replace(" ", "_").replace(".", "_")

def ensure_odd_window(window: int) -> int:
    window = int(window)
    if window < 1:
        return 1
    if window % 2 == 0:
        window += 1
    return window

def fill_inside_nans(x: np.ndarray) -> np.ndarray:
    """Rellena NaN internos solo para operaciones de señal; no modifica los CSV limpios."""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)

def smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Suavizado conservador que tolera NaN."""
    x = np.asarray(x, dtype=float)
    window = ensure_odd_window(window)
    if window <= 1 or len(x) < 3:
        return x.copy()
    y = fill_inside_nans(x)
    if len(y) >= window and window >= 5:
        try:
            poly = 2 if window >= 5 else 1
            return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
        except Exception:
            pass
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)

def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """
    Interpola solo gaps cortos. Gaps largos quedan como NaN.
    """
    return series.interpolate(
        method="linear",
        limit=int(max_gap),
        limit_area="inside",
        limit_direction="both",
    )

def read_dlc_file(path: Path) -> pd.DataFrame:
    """Lee H5 o CSV de DeepLabCut y devuelve DataFrame original."""
    suffix = path.suffix.lower()
    if suffix in [".h5", ".hdf5"]:
        return pd.read_hdf(path)
    if suffix == ".csv":
        # DLC estándar: 3 filas de encabezado, primera columna = frame/index.
        try:
            return pd.read_csv(path, header=[0, 1, 2], index_col=0)
        except Exception:
            pass
        # Respaldo para CSV ya plano.
        return pd.read_csv(path, index_col=0)
    raise ValueError(f"Formato no soportado: {suffix}. Use .h5 o .csv")

def flatten_dlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas DLC MultiIndex a columnas planas tipo bodypart_coord.
    Soporta columnas de 3 niveles: scorer/bodypart/coord.
    """
    out = pd.DataFrame(index=df.index.copy())

    if isinstance(df.columns, pd.MultiIndex):
        nlevels = df.columns.nlevels
        if nlevels < 2:
            raise ValueError("Columnas MultiIndex con menos de 2 niveles; no parece salida DLC estándar.")

        # Buscar nivel de coords por presencia de x/y/likelihood.
        coord_level = None
        for level in range(nlevels):
            vals = set(map(str, df.columns.get_level_values(level)))
            if {"x", "y", "likelihood"}.issubset(vals) or {"x", "y"}.issubset(vals):
                coord_level = level
                break
        if coord_level is None:
            raise ValueError("No pude identificar el nivel de coordenadas x/y/likelihood.")

        # Buscar nivel de bodyparts por coincidencia con BODY_PARTS.
        bp_level = None
        for level in range(nlevels):
            vals = set(map(str, df.columns.get_level_values(level)))
            if any(bp in vals for bp in BODY_PARTS):
                bp_level = level
                break
        if bp_level is None:
            raise ValueError("No pude identificar el nivel de bodyparts.")

        for bp in BODY_PARTS:
            for coord in COORDS:
                matching_cols = [col for col in df.columns if str(col[bp_level]) == bp and str(col[coord_level]) == coord]
                if not matching_cols:
                    continue
                out[f"{bp}_{coord}"] = pd.to_numeric(df[matching_cols[0]], errors="coerce")
    else:
        # Ya plano: intentar reconocer columnas bodypart_coord.
        cols = {str(c): c for c in df.columns}
        for bp in BODY_PARTS:
            for coord in COORDS:
                candidates = [
                    f"{bp}_{coord}",
                    f"{bp}.{coord}",
                    f"{bp} {coord}",
                ]
                for cand in candidates:
                    if cand in cols:
                        out[f"{bp}_{coord}"] = pd.to_numeric(df[cols[cand]], errors="coerce")
                        break

    missing = [f"{bp}_{coord}" for bp in BODY_PARTS for coord in ["x", "y"] if f"{bp}_{coord}" not in out.columns]
    if missing:
        raise ValueError("Faltan columnas necesarias: " + ", ".join(missing))

    # Si algún likelihood no existe, asumir 1.0 para no descartar puntos.
    for bp in BODY_PARTS:
        lh = f"{bp}_likelihood"
        if lh not in out.columns:
            out[lh] = 1.0

    out.index = pd.RangeIndex(start=0, stop=len(out), step=1, name="frame")
    return out

def clean_coordinates(
    flat: pd.DataFrame,
    likelihood_min: float,
    max_gap: int,
    smooth_window: int,
) -> pd.DataFrame:
    """
    Aplica likelihood, interpolación de gaps cortos y suavizado.
    Las columnas finales siguen siendo bodypart_x, bodypart_y, bodypart_likelihood.
    """
    clean = pd.DataFrame(index=flat.index.copy())
    clean.index.name = "frame"

    for bp in BODY_PARTS:
        lh = pd.to_numeric(flat[f"{bp}_likelihood"], errors="coerce")
        clean[f"{bp}_likelihood"] = lh
        good = lh >= float(likelihood_min)

        for coord in ["x", "y"]:
            col = f"{bp}_{coord}"
            s = pd.to_numeric(flat[col], errors="coerce").copy()
            s[~good] = np.nan
            interp = interpolate_short_gaps(s, max_gap=max_gap)
            smooth = smooth_1d(interp.to_numpy(dtype=float), window=smooth_window)
            # Mantener NaN en gaps largos que no se interpolaron.
            smooth[interp.isna().to_numpy()] = np.nan
            clean[col] = smooth

    # Reordenar columnas: bp_x, bp_y, bp_likelihood.
    ordered = []
    for bp in BODY_PARTS:
        ordered += [f"{bp}_x", f"{bp}_y", f"{bp}_likelihood"]
    return clean[ordered]

def calculate_velocity(signal: np.ndarray, fps: float) -> np.ndarray:
    sig = fill_inside_nans(np.asarray(signal, dtype=float))
    if len(sig) < 3:
        return np.full_like(sig, np.nan, dtype=float)
    return np.gradient(sig) * float(fps)

def internal_angle_from_clean(clean: pd.DataFrame, a: str, b: str, c: str) -> np.ndarray:
    """Ángulo interno ABC en grados usando coordenadas limpias."""
    ax = clean[f"{a}_x"].to_numpy(dtype=float)
    ay = clean[f"{a}_y"].to_numpy(dtype=float)
    bx = clean[f"{b}_x"].to_numpy(dtype=float)
    by = clean[f"{b}_y"].to_numpy(dtype=float)
    cx = clean[f"{c}_x"].to_numpy(dtype=float)
    cy = clean[f"{c}_y"].to_numpy(dtype=float)

    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot = v1x * v2x + v1y * v2y
    den = np.hypot(v1x, v1y) * np.hypot(v2x, v2y)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = np.clip(dot / den, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
    ang[(~np.isfinite(den)) | (den <= 0)] = np.nan
    return ang

def detrend_and_standardize_signal(signal: np.ndarray, fps: float) -> np.ndarray:
    """
    Elimina deriva lenta y estandariza de forma robusta.

    La deriva se estima con mediana móvil de ~1 s. Después se aplica un suavizado
    corto (~0.06 s), para no borrar ciclos rápidos en videos de 30-60 fps.
    """
    x = fill_inside_nans(np.asarray(signal, dtype=float))
    if len(x) < 3:
        return np.asarray(x, dtype=float)

    baseline_window = ensure_odd_window(max(5, int(round(float(fps) * 1.0))))
    baseline = pd.Series(x).rolling(
        window=baseline_window, center=True, min_periods=1
    ).median().to_numpy(dtype=float)
    y = x - baseline

    short_window = ensure_odd_window(max(3, int(round(float(fps) * 0.06))))
    if len(y) >= short_window and short_window >= 5:
        try:
            y = savgol_filter(y, window_length=short_window, polyorder=2, mode="interp")
        except Exception:
            pass
    elif short_window > 1:
        y = pd.Series(y).rolling(
            window=short_window, center=True, min_periods=1
        ).mean().to_numpy(dtype=float)

    med = float(np.nanmedian(y))
    mad = float(np.nanmedian(np.abs(y - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanstd(y))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return (y - med) / scale

def _find_extrema_z(
    z: np.ndarray,
    polarity: str,
    distance_frames: int,
    prominence_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extremos en una señal estandarizada robustamente."""
    work = np.asarray(z, dtype=float) if polarity == "max" else -np.asarray(z, dtype=float)
    peaks, props = find_peaks(
        work,
        distance=max(1, int(distance_frames)),
        prominence=max(0.0, float(prominence_z)),
    )
    proms = props.get("prominences", np.full(len(peaks), np.nan, dtype=float))
    return peaks.astype(int), np.asarray(proms, dtype=float)

def detect_multisignal_events(
    clean: pd.DataFrame,
    contact_bodyparts: List[str],
    rhythm_bodyparts: List[str],
    fps: float,
    event_method: str,
    event_polarity: str,
    min_cycle_s: float,
    max_cycle_s: float,
    prominence_px: Optional[float],
    merge_tolerance_frames: int,
    min_event_support: int,
    smooth_window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float, Optional[float], str]:
    """
    Detector multiseñal P30 V2.

    Cada evento debe existir primero como mínimo de Toe X relativo al Hip.
    Un mínimo del ángulo de rodilla hip-knee-ankle es confirmación fuerte.
    Si el ángulo de rodilla no confirma, se exige simultáneamente soporte de
    Knee X relativo al Hip y del ángulo de cadera crest-hip-knee.

    No se crean eventos para rellenar huecos y no se fuerza ningún número de ciclos.
    """
    if fps <= 0:
        raise ValueError("fps debe ser > 0")

    required = [
        "toe_x", "hip_x", "knee_x",
        "crest_x", "crest_y", "hip_y", "knee_y",
        "ankle_x", "ankle_y",
    ]
    missing = [c for c in required if c not in clean.columns]
    if missing:
        raise ValueError("Faltan columnas para detección multiseñal V2: " + ", ".join(missing))

    toe_rel_x = clean["toe_x"].to_numpy(dtype=float) - clean["hip_x"].to_numpy(dtype=float)
    knee_rel_x = clean["knee_x"].to_numpy(dtype=float) - clean["hip_x"].to_numpy(dtype=float)
    hip_angle = internal_angle_from_clean(clean, "crest", "hip", "knee")
    knee_angle = internal_angle_from_clean(clean, "hip", "knee", "ankle")

    toe_z = detrend_and_standardize_signal(toe_rel_x, fps=fps)
    knee_rel_z = detrend_and_standardize_signal(knee_rel_x, fps=fps)
    hip_angle_z = detrend_and_standardize_signal(hip_angle, fps=fps)
    knee_angle_z = detrend_and_standardize_signal(knee_angle, fps=fps)

    selected_polarity = "min" if event_polarity in {"min", "auto"} else event_polarity
    if selected_polarity != "min":
        raise ValueError("La versión multiseñal P30 V2 está estandarizada a event_polarity='min'.")

    # Cinco frames = 83 ms a 60 Hz. Es un límite de resolución, no una cadencia objetivo.
    min_distance_frames = max(5, int(round(float(min_cycle_s) * float(fps))))
    toe_prominence_z = 0.10 if prominence_px is None else float(prominence_px)
    secondary_prominence_z = max(0.10, toe_prominence_z)

    toe_candidates, toe_prom = _find_extrema_z(
        toe_z, polarity="min",
        distance_frames=min_distance_frames,
        prominence_z=toe_prominence_z,
    )

    secondary_distance = max(4, min_distance_frames)
    knee_angle_min, _ = _find_extrema_z(
        knee_angle_z, "min", secondary_distance, secondary_prominence_z
    )
    knee_rel_min, _ = _find_extrema_z(
        knee_rel_z, "min", secondary_distance, secondary_prominence_z
    )
    knee_rel_max, _ = _find_extrema_z(
        knee_rel_z, "max", secondary_distance, secondary_prominence_z
    )
    hip_min, _ = _find_extrema_z(
        hip_angle_z, "min", secondary_distance, secondary_prominence_z
    )
    hip_max, _ = _find_extrema_z(
        hip_angle_z, "max", secondary_distance, secondary_prominence_z
    )
    knee_rel_extrema = np.sort(np.unique(np.concatenate([knee_rel_min, knee_rel_max])))
    hip_extrema = np.sort(np.unique(np.concatenate([hip_min, hip_max])))

    # ±3 frames = ±50 ms.
    tolerance = max(1, int(round(0.05 * float(fps))))

    def nearest_delta(frames: np.ndarray, frame: int) -> int:
        if len(frames) == 0:
            return 10**9
        return int(np.min(np.abs(frames - int(frame))))

    rows = []
    rejected_rows = []
    for frame, prom in zip(toe_candidates, toe_prom):
        frame = int(frame)
        knee_angle_delta = nearest_delta(knee_angle_min, frame)
        knee_rel_delta = nearest_delta(knee_rel_extrema, frame)
        hip_delta = nearest_delta(hip_extrema, frame)

        knee_angle_ok = knee_angle_delta <= tolerance
        knee_rel_ok = knee_rel_delta <= tolerance
        hip_ok = hip_delta <= tolerance

        # A) Toe + mínimo del ángulo de rodilla.
        # B) Toe + Knee relativo + ángulo de cadera.
        accepted = bool(knee_angle_ok or (knee_rel_ok and hip_ok))

        support_count = 1 + int(knee_angle_ok) + int(knee_rel_ok) + int(hip_ok)
        if knee_angle_ok:
            confidence_tier = "A_toe+knee_angle"
        elif knee_rel_ok and hip_ok:
            confidence_tier = "B_toe+knee_rel+hip_angle"
        else:
            confidence_tier = "REJECT_no_secondary_consensus"

        row = {
            "frame": frame,
            "support_count": support_count,
            "confidence_tier": confidence_tier,
            "toe_prominence_z": float(prom) if np.isfinite(prom) else np.nan,
            "knee_angle_delta_frames": knee_angle_delta if knee_angle_delta < 10**8 else np.nan,
            "knee_rel_delta_frames": knee_rel_delta if knee_rel_delta < 10**8 else np.nan,
            "hip_angle_delta_frames": hip_delta if hip_delta < 10**8 else np.nan,
            "knee_angle_support": int(knee_angle_ok),
            "knee_rel_support": int(knee_rel_ok),
            "hip_angle_support": int(hip_ok),
            "event_signal_value": float(toe_z[frame]),
        }
        if accepted:
            rows.append(row)
        else:
            rejected_rows.append(row)

    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values("frame").reset_index(drop=True)
        frames = events["frame"].to_numpy(dtype=int)
        intervals = np.diff(frames).astype(float)
        expected_period = float(np.nanmedian(intervals)) if len(intervals) else np.nan

        prev_interval = np.r_[np.nan, intervals]
        next_interval = np.r_[intervals, np.nan]
        events["interval_from_prev_frames"] = prev_interval
        events["interval_to_next_frames"] = next_interval

        if np.isfinite(expected_period) and expected_period > 0:
            events["review_long_interval_after"] = (
                pd.Series(next_interval).gt(1.60 * expected_period).fillna(False).astype(int)
            )
            events["review_short_interval_after"] = (
                pd.Series(next_interval).lt(0.65 * expected_period).fillna(False).astype(int)
            )
        else:
            events["review_long_interval_after"] = 0
            events["review_short_interval_after"] = 0

        events.insert(0, "event_id", np.arange(1, len(events) + 1, dtype=int))
        events["time_s_60Hz"] = events["frame"] / float(fps)
        events["event_method"] = "multisignal_v2_toe_kneeangle_flexible"
        events["event_coord"] = "toe_x-hip_x + knee_angle + knee_x-hip_x + hip_angle"
        events["event_polarity"] = "min"
        events["support_tolerance_frames"] = tolerance
        events["expected_period_frames"] = expected_period
        events["expected_period_s_60Hz"] = expected_period / float(fps) if np.isfinite(expected_period) else np.nan
    else:
        expected_period = None

    debug = pd.DataFrame(index=clean.index.copy())
    debug.index.name = "frame"
    debug["toe_rel_x"] = toe_rel_x
    debug["knee_rel_x"] = knee_rel_x
    debug["hip_angle_deg"] = hip_angle
    debug["knee_angle_deg"] = knee_angle
    debug["toe_rel_x_z"] = toe_z
    debug["knee_rel_x_z"] = knee_rel_z
    debug["hip_angle_z"] = hip_angle_z
    debug["knee_angle_z"] = knee_angle_z
    debug["distal_x"] = toe_rel_x
    debug["event_signal"] = toe_z
    debug["distal_vx_px_s"] = calculate_velocity(toe_rel_x, fps=fps)

    for col in [
        "candidate_toe_min", "candidate_knee_angle_min",
        "candidate_knee_rel_extremum", "candidate_hip_angle_extremum",
        "selected_event",
    ]:
        debug[col] = 0
    debug.loc[debug.index.intersection(toe_candidates), "candidate_toe_min"] = 1
    debug.loc[debug.index.intersection(knee_angle_min), "candidate_knee_angle_min"] = 1
    debug.loc[debug.index.intersection(knee_rel_extrema), "candidate_knee_rel_extremum"] = 1
    debug.loc[debug.index.intersection(hip_extrema), "candidate_hip_angle_extremum"] = 1
    if not events.empty:
        debug.loc[debug.index.intersection(events["frame"].to_numpy(dtype=int)), "selected_event"] = 1

    debug["candidate_min_event"] = debug["candidate_toe_min"]
    debug["candidate_max_event"] = 0
    debug.attrs["rejected_candidates"] = rejected_rows

    return events, debug, toe_z, toe_prominence_z, expected_period, "min"

def build_cycles_from_events(
    events: pd.DataFrame,
    clean: pd.DataFrame,
    event_signal: np.ndarray,
    fps: float,
    min_cycle_s: float,
    max_cycle_s: float,
    expected_period_frames: Optional[float],
) -> pd.DataFrame:
    """Construye ciclos entre eventos consecutivos. Solo rechaza duración extrema."""
    columns = [
        "cycle_id", "start_event_id", "end_event_id", "start_frame", "end_frame",
        "start_time_s", "end_time_s", "duration_frames", "duration_s",
        "expected_period_frames", "duration_ratio_to_expected",
        "start_support", "end_support", "mean_event_support",
        "valid_fraction_event_signal", "nan_fraction_event_signal", "event_signal_range_px",
        "accepted", "reject_reason",
    ]

    if events.empty or len(events) < 2:
        return pd.DataFrame(columns=columns)

    min_frames = max(1, int(round(min_cycle_s * fps)))
    max_frames = max(min_frames + 1, int(round(max_cycle_s * fps)))

    frames = events["frame"].to_numpy(dtype=int)
    event_ids = events["event_id"].to_numpy(dtype=int)
    supports = events["support_count"].to_numpy(dtype=float) if "support_count" in events.columns else np.ones(len(events))

    rows = []
    cycle_id = 0
    for i in range(len(frames) - 1):
        start = int(frames[i])
        end = int(frames[i + 1])
        duration_frames = end - start
        duration_s = duration_frames / float(fps)

        seg = event_signal[start:end + 1]
        finite = np.isfinite(seg)
        valid_fraction = float(finite.mean()) if len(seg) else 0.0
        nan_fraction = 1.0 - valid_fraction
        signal_range = float(np.nanmax(seg) - np.nanmin(seg)) if finite.any() else np.nan

        accepted = True
        reasons = []
        if duration_frames < min_frames:
            accepted = False
            reasons.append("duracion_menor_al_minimo")
        if duration_frames > max_frames:
            accepted = False
            reasons.append("duracion_mayor_al_maximo")

        cycle_id += 1
        ratio = np.nan
        if expected_period_frames is not None and np.isfinite(expected_period_frames) and expected_period_frames > 0:
            ratio = float(duration_frames / expected_period_frames)

        rows.append({
            "cycle_id": cycle_id,
            "start_event_id": int(event_ids[i]),
            "end_event_id": int(event_ids[i + 1]),
            "start_frame": start,
            "end_frame": end,
            "start_time_s": start / float(fps),
            "end_time_s": end / float(fps),
            "duration_frames": int(duration_frames),
            "duration_s": float(duration_s),
            "expected_period_frames": expected_period_frames if expected_period_frames is not None else np.nan,
            "duration_ratio_to_expected": ratio,
            "start_support": float(supports[i]),
            "end_support": float(supports[i + 1]),
            "mean_event_support": float(np.nanmean([supports[i], supports[i + 1]])),
            "valid_fraction_event_signal": valid_fraction,
            "nan_fraction_event_signal": nan_fraction,
            "event_signal_range_px": signal_range,
            "accepted": int(accepted),
            "reject_reason": ";".join(reasons) if reasons else "",
        })

    return pd.DataFrame(rows, columns=columns)

def normalize_cycles(clean: pd.DataFrame, cycles: pd.DataFrame, n_points: int = 101, accepted_only: bool = True) -> pd.DataFrame:
    """Normaliza coordenadas limpias a 0-100 % para cada ciclo."""
    if cycles.empty:
        return pd.DataFrame()
    use_cycles = cycles.copy()
    if accepted_only and "accepted" in use_cycles.columns:
        use_cycles = use_cycles[use_cycles["accepted"] == 1].copy()
    if use_cycles.empty:
        return pd.DataFrame()

    percent = np.linspace(0, 100, int(n_points))
    rows = []
    coord_cols = [f"{bp}_{c}" for bp in BODY_PARTS for c in ["x", "y"]]

    for _, cyc in use_cycles.iterrows():
        start = int(cyc["start_frame"])
        end = int(cyc["end_frame"])
        if end <= start:
            continue
        frame_idx = np.arange(start, end + 1, dtype=int)
        t_old = np.linspace(0, 100, len(frame_idx))

        block: Dict[str, np.ndarray] = {
            "cycle_id": np.full(int(n_points), int(cyc["cycle_id"]), dtype=int),
            "percent_gait_cycle": percent,
            "start_frame": np.full(int(n_points), start, dtype=int),
            "end_frame": np.full(int(n_points), end, dtype=int),
            "duration_frames": np.full(int(n_points), int(cyc["duration_frames"]), dtype=int),
            "duration_s": np.full(int(n_points), float(cyc["duration_s"]), dtype=float),
        }

        for col in coord_cols:
            values = clean.loc[frame_idx, col].to_numpy(dtype=float)
            if np.isfinite(values).sum() < 2:
                block[col] = np.full(int(n_points), np.nan, dtype=float)
            else:
                valid = np.isfinite(values)
                block[col] = np.interp(percent, t_old[valid], values[valid])

        rows.append(pd.DataFrame(block))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def plot_cycle_detection_check(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    cycles: pd.DataFrame,
    debug: pd.DataFrame,
    out_png: Path,
    fps: float,
    event_method: str,
    selected_polarity: str,
) -> None:
    """Gráfico obligatorio para verificar visualmente el conteo de ciclos."""
    frames = clean.index.to_numpy(dtype=int)
    t = frames / float(fps)

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)

    # 1) distal x/y y eventos.
    ax = axes[0]
    if "distal_x" in debug.columns:
        ax.plot(t, debug["distal_x"], linewidth=1.2, label="Toe X relativo a Hip")
    if "distal_y" in debug.columns:
        ax.plot(t, debug["distal_y"], linewidth=1.0, alpha=0.75, label="distal_y")
    if not events.empty:
        ev_t = events["frame"].to_numpy(dtype=int) / float(fps)
        y = debug.loc[events["frame"].to_numpy(dtype=int), "distal_x"].to_numpy(dtype=float) if "distal_x" in debug.columns else debug.loc[events["frame"].to_numpy(dtype=int), "event_signal"].to_numpy(dtype=float)
        ax.scatter(ev_t, y, s=28, marker="o", label="eventos aceptados")
    ax.set_ylabel("Toe X - Hip X (px)")
    ax.set_title("Detección multiseñal validada: Toe + ángulo de rodilla + respaldo Knee/cadera (60 Hz)")
    ax.legend(loc="best")

    # 2) consenso de las tres señales estandarizadas.
    ax = axes[1]
    if "toe_rel_x_z" in debug.columns:
        ax.plot(t, debug["toe_rel_x_z"], linewidth=1.2, label="Toe rel. X (z)")
    if "knee_rel_x_z" in debug.columns:
        ax.plot(t, debug["knee_rel_x_z"], linewidth=1.0, alpha=0.80, label="Knee rel. X (z)")
    if "hip_angle_z" in debug.columns:
        ax.plot(t, debug["hip_angle_z"], linewidth=1.0, alpha=0.75, label="Ángulo cadera (z)")
    if "candidate_min_event" in debug.columns:
        idx = debug.index[debug["candidate_min_event"] == 1].to_numpy(dtype=int)
        if len(idx):
            ax.scatter(idx / float(fps), debug.loc[idx, "toe_rel_x_z"], s=18, marker="v", label="candidatos Toe")
    if not events.empty:
        idx = events["frame"].to_numpy(dtype=int)
        ax.scatter(idx / float(fps), debug.loc[idx, "toe_rel_x_z"], s=34, marker="o", label="eventos multiseñal validados")
    ax.set_ylabel("señales robustas (z)")
    ax.legend(loc="best", ncol=2)

    # 3) velocidad horizontal distal.
    ax = axes[2]
    if "distal_vx_px_s" in debug.columns:
        ax.plot(t, debug["distal_vx_px_s"], linewidth=1.0, label="distal_vx")
        ax.axhline(0, linewidth=0.8)
    if not events.empty:
        idx = events["frame"].to_numpy(dtype=int)
        ax.scatter(idx / float(fps), debug.loc[idx, "distal_vx_px_s"], s=25, marker="o", label="eventos")
    ax.set_ylabel("velocidad x (px/s)")
    ax.legend(loc="best")

    # 4) barras de ciclos aceptados/rechazados.
    ax = axes[3]
    if "toe_y" in clean.columns:
        ax.plot(t, clean["toe_y"], linewidth=0.9, label="toe_y")
    elif "foot_y" in clean.columns:
        ax.plot(t, clean["foot_y"], linewidth=0.9, label="foot_y")

    if not cycles.empty:
        ymin, ymax = ax.get_ylim()
        for _, row in cycles.iterrows():
            start_t = float(row["start_frame"]) / float(fps)
            end_t = float(row["end_frame"]) / float(fps)
            if int(row.get("accepted", 1)) == 1:
                ax.axvspan(start_t, end_t, alpha=0.15)
            else:
                ax.axvspan(start_t, end_t, alpha=0.08, hatch="//")
        ax.set_ylim(ymin, ymax)
    ax.set_ylabel("toe_y (px)")
    ax.set_xlabel("tiempo (s)")
    ax.legend(loc="best")

    n_events = 0 if events.empty else len(events)
    n_acc = 0 if cycles.empty else int((cycles["accepted"] == 1).sum())
    n_rej = 0 if cycles.empty else int((cycles["accepted"] != 1).sum())
    fig.suptitle(f"Eventos: {n_events} | Ciclos aceptados: {n_acc} | Rechazados: {n_rej}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

def write_params(
    params_file: Path,
    input_file: Path,
    fps: float,
    likelihood_min: float,
    max_gap: int,
    analysis_smooth_window: int,
    detection_smooth_window: int,
    min_cycle_s: float,
    max_cycle_s: float,
    prominence_z: float,
    expected_period: Optional[float],
    n_events: int,
    n_cycles: int,
    n_accepted: int,
    treadmill_speed_cm_s: float,
    sex: str,
    litter_id: str,
) -> None:
    """Guarda un registro reproducible de la ejecución."""
    lines = [
        f"script = 01_preprocesamiento_y_ciclos.py",
        f"algorithm_version = {ALGORITHM_VERSION}",
        f"input_file = {input_file}",
        f"fps = {fps}",
        f"likelihood_min = {likelihood_min}",
        f"max_gap_interpolation = {max_gap}",
        f"analysis_smooth_window = {analysis_smooth_window}",
        f"detection_smooth_window = {detection_smooth_window}",
        f"min_cycle_duration_s = {min_cycle_s}",
        f"max_cycle_duration_s = {max_cycle_s}",
        f"toe_candidate_prominence_z = {prominence_z}",
        f"expected_period_frames = {expected_period if expected_period is not None else 'NA'}",
        f"expected_period_s = {(expected_period / fps) if expected_period is not None and np.isfinite(expected_period) else 'NA'}",
        f"n_events = {n_events}",
        f"n_cycles_total = {n_cycles}",
        f"n_cycles_accepted = {n_accepted}",
        f"treadmill_speed_cm_s = {treadmill_speed_cm_s}",
        f"sex = {sex}",
        f"litter_id = {litter_id}",
        "event_primary = local minimum of toe_x - hip_x",
        "event_confirmation_A = nearby minimum of knee angle hip-knee-ankle",
        "event_confirmation_B = nearby knee_x-hip_x extremum AND nearby hip angle crest-hip-knee extremum",
        "synthetic_events = NEVER",
        "manual_validation = detector logic visually validated on all 10 P30 videos",
    ]
    params_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    input_file: Path,
    outdir: Path,
    fps: float = FPS,
    likelihood_min: float = LIKELIHOOD_MIN,
    max_gap: int = MAX_GAP_INTERPOLATION,
    analysis_smooth_window: int = ANALYSIS_SMOOTH_WINDOW,
    detection_smooth_window: int = DETECTION_SMOOTH_WINDOW,
    min_cycle_s: float = MIN_CYCLE_DURATION_S,
    max_cycle_s: float = MAX_CYCLE_DURATION_S,
    prominence_z: float = DEFAULT_PROMINENCE_Z,
    norm_points: int = NORM_POINTS,
    make_plot: bool = True,
    treadmill_speed_cm_s: float = DEFAULT_TREADMILL_SPEED_CM_S,
    sex: str = DEFAULT_SEX,
    litter_id: str = DEFAULT_LITTER_ID,
) -> Dict[str, Path]:
    """Ejecuta limpieza, detección multiseñal, construcción de ciclos y exportación."""
    input_file = Path(input_file)
    outdir = Path(outdir)
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_file}")
    if fps <= 0:
        raise ValueError("fps debe ser > 0")
    if abs(float(fps) - 60.0) > 1e-9:
        print(f"ADVERTENCIA: esta versión fue validada a 60 Hz; se solicitó fps={fps}.")

    outdir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_stem(input_file)

    raw = read_dlc_file(input_file)
    flat = flatten_dlc_columns(raw)

    clean = clean_coordinates(
        flat=flat,
        likelihood_min=likelihood_min,
        max_gap=max_gap,
        smooth_window=analysis_smooth_window,
    )
    if detection_smooth_window == analysis_smooth_window:
        detection_clean = clean
    else:
        detection_clean = clean_coordinates(
            flat=flat,
            likelihood_min=likelihood_min,
            max_gap=max_gap,
            smooth_window=detection_smooth_window,
        )

    events, debug, event_signal, used_prominence, expected_period, selected_polarity = detect_multisignal_events(
        clean=detection_clean,
        contact_bodyparts=["toe"],
        rhythm_bodyparts=["knee", "hip"],
        fps=fps,
        event_method="multisignal_v2",
        event_polarity="min",
        min_cycle_s=min_cycle_s,
        max_cycle_s=max_cycle_s,
        prominence_px=prominence_z,
        merge_tolerance_frames=3,
        min_event_support=2,
        smooth_window=detection_smooth_window,
    )

    cycles = build_cycles_from_events(
        events=events,
        clean=clean,
        event_signal=event_signal,
        fps=fps,
        min_cycle_s=min_cycle_s,
        max_cycle_s=max_cycle_s,
        expected_period_frames=expected_period,
    )
    normalized = normalize_cycles(clean, cycles, n_points=norm_points, accepted_only=True)

    clean_csv = outdir / f"{stem}_clean_coords.csv"
    events_csv = outdir / f"{stem}_events_detected.csv"
    cycles_csv = outdir / f"{stem}_gait_cycles.csv"
    norm_csv = outdir / f"{stem}_normalized_cycles.csv"
    debug_csv = outdir / f"{stem}_cycle_detection_signals.csv"
    rejected_csv = outdir / f"{stem}_rejected_event_candidates.csv"
    check_png = outdir / f"{stem}_cycle_detection_check.png"
    params_txt = outdir / f"{stem}_params.txt"

    clean.to_csv(clean_csv, index=True)
    events.to_csv(events_csv, index=False)
    cycles.to_csv(cycles_csv, index=False)
    normalized.to_csv(norm_csv, index=False)
    debug.to_csv(debug_csv, index=True)
    pd.DataFrame(debug.attrs.get("rejected_candidates", [])).to_csv(rejected_csv, index=False)

    if make_plot:
        plot_cycle_detection_check(
            clean=clean,
            events=events,
            cycles=cycles,
            debug=debug,
            out_png=check_png,
            fps=fps,
            event_method="multisignal_v2",
            selected_polarity=selected_polarity,
        )

    n_events = len(events)
    n_cycles = len(cycles)
    n_accepted = int((cycles["accepted"] == 1).sum()) if not cycles.empty else 0
    write_params(
        params_file=params_txt,
        input_file=input_file,
        fps=fps,
        likelihood_min=likelihood_min,
        max_gap=max_gap,
        analysis_smooth_window=analysis_smooth_window,
        detection_smooth_window=detection_smooth_window,
        min_cycle_s=min_cycle_s,
        max_cycle_s=max_cycle_s,
        prominence_z=used_prominence,
        expected_period=expected_period,
        n_events=n_events,
        n_cycles=n_cycles,
        n_accepted=n_accepted,
        treadmill_speed_cm_s=treadmill_speed_cm_s,
        sex=sex,
        litter_id=litter_id,
    )

    print(f"{stem}: {n_events} eventos -> {n_cycles} ciclos ({n_accepted} aceptados)")
    return {
        "clean_coords": clean_csv,
        "events_detected": events_csv,
        "gait_cycles": cycles_csv,
        "normalized_cycles": norm_csv,
        "cycle_detection_signals": debug_csv,
        "rejected_event_candidates": rejected_csv,
        "cycle_detection_check": check_png,
        "params": params_txt,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Detecta ciclos de marcha P30 con el algoritmo multiseñal validado: "
            "Toe relativo + ángulo de rodilla, con respaldo Knee/cadera."
        )
    )
    p.add_argument("input_file", type=Path, help="Archivo DeepLabCut .h5/.hdf5 o .csv")
    p.add_argument("--outdir", type=Path, default=Path("salida_01_ciclos"))
    p.add_argument("--fps", type=float, default=FPS)
    p.add_argument("--likelihood-min", type=float, default=LIKELIHOOD_MIN)
    p.add_argument("--max-gap", type=int, default=MAX_GAP_INTERPOLATION)
    p.add_argument("--analysis-smooth-window", type=int, default=ANALYSIS_SMOOTH_WINDOW)
    p.add_argument("--detection-smooth-window", type=int, default=DETECTION_SMOOTH_WINDOW)
    p.add_argument("--min-cycle-s", type=float, default=MIN_CYCLE_DURATION_S)
    p.add_argument("--max-cycle-s", type=float, default=MAX_CYCLE_DURATION_S)
    p.add_argument("--prominence-z", type=float, default=DEFAULT_PROMINENCE_Z)
    p.add_argument("--norm-points", type=int, default=NORM_POINTS)
    p.add_argument("--treadmill-speed-cm-s", type=float, default=DEFAULT_TREADMILL_SPEED_CM_S,
                   help="Metadato experimental. P30 Day 1: 20 cm/s.")
    p.add_argument("--sex", type=str, default=DEFAULT_SEX,
                   help="Metadato experimental. P30 Day 1: male.")
    p.add_argument("--litter-id", type=str, default=DEFAULT_LITTER_ID,
                   help="Metadato experimental. P30 Day 1: una sola camada compartida.")
    p.add_argument("--no-plot", action="store_true", help="No generar PNG de control")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_pipeline(
            input_file=args.input_file,
            outdir=args.outdir,
            fps=args.fps,
            likelihood_min=args.likelihood_min,
            max_gap=args.max_gap,
            analysis_smooth_window=args.analysis_smooth_window,
            detection_smooth_window=args.detection_smooth_window,
            min_cycle_s=args.min_cycle_s,
            max_cycle_s=args.max_cycle_s,
            prominence_z=args.prominence_z,
            norm_points=args.norm_points,
            make_plot=not args.no_plot,
            treadmill_speed_cm_s=args.treadmill_speed_cm_s,
            sex=args.sex,
            litter_id=args.litter_id,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
