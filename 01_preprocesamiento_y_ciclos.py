#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_preprocesamiento_y_ciclos.py

Primera parte del pipeline para análisis de marcha murina con DeepLabCut.

Este script SOLO hace:
1. Leer archivo .h5 o .csv de DeepLabCut.
2. Extraer crest, hip, knee, ankle, foot y toe.
3. Filtrar coordenadas por likelihood.
4. Interpolar gaps cortos.
5. Suavizar coordenadas.
6. Detectar ciclos de marcha.
7. Normalizar cada ciclo aceptado a 0-100%.
8. Exportar archivos de control.

Versión actual:
- Método principal: distal_x.
- Usa toe + foot como señal distal combinada.
- Detecta eventos por extremos de la posición horizontal distal y velocidad.
- hip/knee/ankle NO definen contacto; se exportan como señales auxiliares de ritmo.
- Rechazo mínimo de ciclos: solo duración fuera del rango fisiológico configurado.

Uso recomendado:
    python 01_preprocesamiento_y_ciclos.py "archivo_filtrado.h5" --fps 30 --outdir salida_01_ciclos

Si la cámara está invertida o el inicio de ciclo queda en toe-off en vez de foot strike:
    python 01_preprocesamiento_y_ciclos.py "archivo_filtrado.h5" --event-polarity min

Opciones útiles:
    --event-method distal_x        método recomendado
    --event-polarity auto|max|min  auto intenta inferir foot strike
    --prominence 3                 baja/sube sensibilidad
    --contact-bodyparts toe,foot   puntos distales usados
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks, savgol_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================================================================
# PARAMETROS EDITABLES
# =============================================================================

FPS = 30.0

BODY_PARTS = ["crest", "hip", "knee", "ankle", "foot", "toe"]
COORDS = ["x", "y", "likelihood"]

# Tracking
LIKELIHOOD_MIN = 0.80
MAX_GAP_INTERPOLATION = 5
SMOOTH_WINDOW = 5

# Detección de ciclos
CONTACT_BODY_PARTS = ["toe", "foot"]
RHYTHM_BODY_PARTS = ["ankle", "knee", "hip"]

# Método recomendado para treadmill lateral: extremos de posición horizontal distal.
EVENT_METHOD = "distal_x"      # distal_x, distal_y, velocity_x
EVENT_POLARITY = "auto"        # auto, max, min

# Con 30 FPS cada frame = 33,3 ms.
# Mantener estos rangos amplios evita eliminar ciclos reales; ajustar si el gráfico lo exige.
MIN_CYCLE_DURATION_S = 0.12
MAX_CYCLE_DURATION_S = 1.50

# Si None, se calcula automático con la variabilidad de la señal.
DETECTION_PROMINENCE_PX = None
MIN_PROMINENCE_PX = 2.0
PROMINENCE_SD_FACTOR = 0.20

# Para agrupar eventos de toe y foot que caen casi en el mismo frame.
MERGE_TOLERANCE_FRAMES = 2
MIN_EVENT_SUPPORT = 1

# Normalización 0-100 % del ciclo.
NORM_POINTS = 101

# No rechazar por NaN/rango; solo reportar calidad.
ACCEPT_ONLY_DURATION = True


# =============================================================================
# UTILIDADES GENERALES
# =============================================================================

def sanitize_stem(path: Path) -> str:
    """Genera un stem seguro para nombres de salida."""
    return path.stem.replace(" ", "_").replace(".", "_")


def parse_bodypart_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in value if str(x).strip()]
    return parts


def ensure_odd_window(window: int) -> int:
    window = int(window)
    if window < 1:
        return 1
    if window % 2 == 0:
        window += 1
    return window


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            return np.zeros_like(x, dtype=float)
        return (x - np.nanmean(x)) / sd
    return 0.6745 * (x - med) / mad


def fill_inside_nans(x: np.ndarray) -> np.ndarray:
    """Rellena NaN internos solo para operaciones de señal; no modifica los CSV limpios."""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    window = ensure_odd_window(window)
    if window <= 1:
        return np.asarray(x, dtype=float)
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Suavizado conservador que tolera NaN."""
    x = np.asarray(x, dtype=float)
    window = ensure_odd_window(window)
    if window <= 1 or len(x) < 3:
        return x.copy()
    y = fill_inside_nans(x)
    if SCIPY_AVAILABLE and len(y) >= window and window >= 5:
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


# =============================================================================
# LECTURA DE DEEPLABCUT
# =============================================================================

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


# =============================================================================
# LIMPIEZA DE COORDENADAS
# =============================================================================

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


# =============================================================================
# DETECCION DE CICLOS
# =============================================================================

def make_distal_signal(
    clean: pd.DataFrame,
    bodyparts: List[str],
    coord: str,
    smooth_window: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Crea señal distal combinada desde toe/foot.
    Se promedia cada punto distal en pixeles, manteniendo la escala original.
    """
    signals = []
    debug_cols: Dict[str, np.ndarray] = {}

    for bp in bodyparts:
        col = f"{bp}_{coord}"
        if col not in clean.columns:
            raise ValueError(f"No existe la columna {col}")
        sig = clean[col].to_numpy(dtype=float)
        sig_filled = fill_inside_nans(sig)
        sig_smooth = smooth_1d(sig_filled, window=max(3, smooth_window))
        signals.append(sig_smooth)
        debug_cols[f"{bp}_{coord}"] = sig_smooth

    if not signals:
        raise ValueError("No hay bodyparts distales para construir la señal.")

    stack = np.vstack(signals)
    combined = np.nanmean(stack, axis=0)
    combined = smooth_1d(combined, window=max(3, smooth_window))
    debug_cols[f"distal_{coord}"] = combined

    debug = pd.DataFrame(debug_cols, index=clean.index.copy())
    debug.index.name = "frame"
    return combined, debug


def calculate_velocity(signal: np.ndarray, fps: float) -> np.ndarray:
    sig = fill_inside_nans(np.asarray(signal, dtype=float))
    if len(sig) < 3:
        return np.full_like(sig, np.nan, dtype=float)
    return np.gradient(sig) * float(fps)


def auto_prominence(signal: np.ndarray) -> float:
    sig = np.asarray(signal, dtype=float)
    spread = np.nanpercentile(sig, 95) - np.nanpercentile(sig, 5)
    sd = np.nanstd(sig)
    val = max(MIN_PROMINENCE_PX, float(PROMINENCE_SD_FACTOR * sd), float(0.08 * spread))
    if not np.isfinite(val) or val <= 0:
        val = MIN_PROMINENCE_PX
    return float(val)


def estimate_period_autocorr(signal: np.ndarray, fps: float, min_cycle_s: float, max_cycle_s: float) -> Optional[float]:
    """Estima periodo dominante en frames usando autocorrelación."""
    x = fill_inside_nans(np.asarray(signal, dtype=float))
    if len(x) < 10:
        return None
    z = robust_zscore(x)
    z = z - np.nanmean(z)
    z[~np.isfinite(z)] = 0.0

    min_lag = max(2, int(round(min_cycle_s * fps)))
    max_lag = min(len(z) // 2, int(round(max_cycle_s * fps)))
    if max_lag <= min_lag:
        return None

    ac = np.correlate(z, z, mode="full")
    ac = ac[len(ac) // 2:]
    segment = ac[min_lag:max_lag + 1]
    if len(segment) == 0 or not np.isfinite(segment).any():
        return None
    lag = int(np.nanargmax(segment) + min_lag)
    if min_lag <= lag <= max_lag:
        return float(lag)
    return None


def find_extrema_events(
    signal: np.ndarray,
    fps: float,
    min_cycle_s: float,
    max_cycle_s: float,
    prominence: float,
    polarity: str,
    expected_period: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Detecta máximos o mínimos locales de la señal."""
    if not SCIPY_AVAILABLE:
        raise RuntimeError("Este script requiere scipy para detección robusta de ciclos. Instala scipy o ejecuta en entorno con scipy.")

    sig = fill_inside_nans(np.asarray(signal, dtype=float))
    min_frames = max(1, int(round(min_cycle_s * fps)))
    if expected_period is not None and np.isfinite(expected_period):
        # Separación entre eventos del mismo tipo. 0.55 evita duplicados y tolera variación real.
        distance = max(min_frames, int(round(0.55 * expected_period)))
    else:
        distance = min_frames

    work = sig if polarity == "max" else -sig
    peaks, props = find_peaks(work, distance=distance, prominence=prominence)
    proms = props.get("prominences", np.full(len(peaks), np.nan, dtype=float))
    return peaks.astype(int), np.asarray(proms, dtype=float)


def interval_quality(frames: np.ndarray, fps: float, min_cycle_s: float, max_cycle_s: float) -> Dict[str, float]:
    """Métrica simple para comparar trenes de eventos."""
    frames = np.asarray(frames, dtype=int)
    if len(frames) < 3:
        return {"n": float(len(frames)), "valid_n": 0.0, "median_interval": np.nan, "cv": np.inf, "score": -np.inf}
    intervals = np.diff(frames).astype(float)
    min_f = max(1, int(round(min_cycle_s * fps)))
    max_f = max(min_f + 1, int(round(max_cycle_s * fps)))
    valid = intervals[(intervals >= min_f) & (intervals <= max_f)]
    if len(valid) < 2:
        return {"n": float(len(frames)), "valid_n": float(len(valid)), "median_interval": np.nan, "cv": np.inf, "score": -np.inf}
    med = float(np.nanmedian(valid))
    cv = float(np.nanstd(valid, ddof=1) / med) if med > 0 and len(valid) > 1 else np.inf
    score = float(len(valid) - 2.0 * cv)
    return {"n": float(len(frames)), "valid_n": float(len(valid)), "median_interval": med, "cv": cv, "score": score}


def infer_foot_strike_polarity_from_halves(
    signal: np.ndarray,
    max_frames: np.ndarray,
    min_frames: np.ndarray,
) -> Optional[str]:
    """
    Intenta inferir si el foot strike corresponde a máximo o mínimo.
    En treadmill lateral, swing suele ser más corto y rápido que stance.
    Si min->max es más corto que max->min, el extremo final de swing es max.
    Si max->min es más corto, el extremo final de swing es min.
    """
    max_frames = np.asarray(max_frames, dtype=int)
    min_frames = np.asarray(min_frames, dtype=int)
    if len(max_frames) < 2 or len(min_frames) < 2:
        return None

    min_to_next_max = []
    max_to_next_min = []

    for mn in min_frames:
        after = max_frames[max_frames > mn]
        if len(after):
            min_to_next_max.append(int(after[0] - mn))
    for mx in max_frames:
        after = min_frames[min_frames > mx]
        if len(after):
            max_to_next_min.append(int(after[0] - mx))

    if len(min_to_next_max) < 2 or len(max_to_next_min) < 2:
        return None

    a = float(np.nanmedian(min_to_next_max))
    b = float(np.nanmedian(max_to_next_min))
    if not np.isfinite(a) or not np.isfinite(b):
        return None

    # Exigir diferencia mínima para no forzar una inferencia dudosa.
    if abs(a - b) < 1.0:
        return None
    return "max" if a < b else "min"


def detect_distal_x_events(
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
    Detecta eventos de marcha usando señal distal combinada.
    Por defecto, distal_x: extremos horizontales de toe+foot.
    """
    contact_bodyparts = parse_bodypart_list(contact_bodyparts)
    rhythm_bodyparts = parse_bodypart_list(rhythm_bodyparts)

    for bp in contact_bodyparts + rhythm_bodyparts:
        if bp not in BODY_PARTS:
            raise ValueError(f"Bodypart no reconocido: {bp}. Opciones: {BODY_PARTS}")

    if event_method == "distal_x":
        coord = "x"
        base_signal, debug = make_distal_signal(clean, contact_bodyparts, coord="x", smooth_window=smooth_window)
        event_signal = base_signal
    elif event_method == "distal_y":
        coord = "y"
        base_signal, debug = make_distal_signal(clean, contact_bodyparts, coord="y", smooth_window=smooth_window)
        event_signal = base_signal
    elif event_method == "velocity_x":
        coord = "x"
        base_signal, debug = make_distal_signal(clean, contact_bodyparts, coord="x", smooth_window=smooth_window)
        vx = calculate_velocity(base_signal, fps=fps)
        # Eventos por picos de desaceleración/cambio de velocidad. Menos recomendado que distal_x.
        event_signal = -np.abs(smooth_1d(vx, window=max(3, smooth_window)))
        debug["distal_vx_px_s"] = vx
        debug["velocity_event_signal"] = event_signal
    else:
        raise ValueError("event_method debe ser distal_x, distal_y o velocity_x")

    # Señales auxiliares para inspección, no para definir contactos.
    for bp in rhythm_bodyparts:
        for c in ["x", "y"]:
            col = f"{bp}_{c}"
            if col in clean.columns:
                debug[f"rhythm_{bp}_{c}"] = smooth_1d(fill_inside_nans(clean[col].to_numpy(dtype=float)), window=max(3, smooth_window))

    if "distal_vx_px_s" not in debug.columns:
        debug["distal_vx_px_s"] = calculate_velocity(base_signal, fps=fps)
    debug["event_signal"] = event_signal

    expected_period = estimate_period_autocorr(event_signal, fps=fps, min_cycle_s=min_cycle_s, max_cycle_s=max_cycle_s)
    prominence = auto_prominence(event_signal) if prominence_px is None else float(prominence_px)

    # Detectar máximos y mínimos. Si auto, elegir por heurística de swing corto; si falla, calidad de intervalos.
    max_events, max_prom = find_extrema_events(
        event_signal, fps=fps, min_cycle_s=min_cycle_s, max_cycle_s=max_cycle_s,
        prominence=prominence, polarity="max", expected_period=expected_period,
    )
    min_events, min_prom = find_extrema_events(
        event_signal, fps=fps, min_cycle_s=min_cycle_s, max_cycle_s=max_cycle_s,
        prominence=prominence, polarity="min", expected_period=expected_period,
    )

    selected_polarity = event_polarity
    if event_polarity == "auto":
        inferred = infer_foot_strike_polarity_from_halves(event_signal, max_events, min_events)
        if inferred is not None:
            selected_polarity = inferred
        else:
            qmax = interval_quality(max_events, fps, min_cycle_s, max_cycle_s)
            qmin = interval_quality(min_events, fps, min_cycle_s, max_cycle_s)
            selected_polarity = "max" if qmax["score"] >= qmin["score"] else "min"

    if selected_polarity == "max":
        selected_events = max_events
        selected_prom = max_prom
    elif selected_polarity == "min":
        selected_events = min_events
        selected_prom = min_prom
    else:
        raise ValueError("event_polarity debe ser auto, max o min")

    # Calcular soporte por toe/foot individuales del mismo tipo.
    support_rows = []
    for frame, prom in zip(selected_events, selected_prom):
        sources = ["distal_combined"]
        source_frames = [int(frame)]
        for bp in contact_bodyparts:
            col = f"{bp}_{coord}"
            if col not in clean.columns:
                continue
            sig_bp = smooth_1d(fill_inside_nans(clean[col].to_numpy(dtype=float)), window=max(3, smooth_window))
            bp_events, _ = find_extrema_events(
                sig_bp, fps=fps, min_cycle_s=min_cycle_s, max_cycle_s=max_cycle_s,
                prominence=max(MIN_PROMINENCE_PX, prominence * 0.60),
                polarity=selected_polarity, expected_period=expected_period,
            )
            close = bp_events[np.abs(bp_events - int(frame)) <= int(merge_tolerance_frames)]
            if len(close):
                sources.append(f"{bp}_{coord}")
                source_frames.append(int(close[np.argmin(np.abs(close - int(frame)))]))

        support_count = len(set(sources))
        if support_count < int(min_event_support):
            continue

        support_rows.append({
            "frame": int(frame),
            "support_count": int(support_count),
            "sources": ";".join(sorted(set(sources))),
            "source_frames": ";".join(map(str, sorted(set(source_frames)))),
            "mean_prominence": float(prom) if np.isfinite(prom) else np.nan,
            "event_signal_value": float(event_signal[int(frame)]),
        })

    events = pd.DataFrame(support_rows).sort_values("frame").reset_index(drop=True)

    # Limpieza conservadora de duplicados por si quedaron eventos demasiado cercanos.
    min_frames = max(1, int(round(min_cycle_s * fps)))
    if not events.empty and len(events) > 1:
        rows = events.to_dict("records")
        cleaned = []
        for row in rows:
            if not cleaned:
                cleaned.append(row)
                continue
            gap = int(row["frame"]) - int(cleaned[-1]["frame"])
            if gap < min_frames:
                prev = cleaned[-1]
                score_prev = (int(prev.get("support_count", 1)), float(prev.get("mean_prominence", 0) or 0), abs(float(prev.get("event_signal_value", 0) or 0)))
                score_new = (int(row.get("support_count", 1)), float(row.get("mean_prominence", 0) or 0), abs(float(row.get("event_signal_value", 0) or 0)))
                if score_new > score_prev:
                    cleaned[-1] = row
            else:
                cleaned.append(row)
        events = pd.DataFrame(cleaned).sort_values("frame").reset_index(drop=True)

    if not events.empty:
        intervals = np.diff(events["frame"].to_numpy(dtype=int)).astype(float)
        period_from_events = float(np.nanmedian(intervals)) if len(intervals) else np.nan
        if np.isfinite(period_from_events):
            expected_period = period_from_events

        events.insert(0, "event_id", np.arange(1, len(events) + 1, dtype=int))
        events["time_s"] = events["frame"] / float(fps)
        events["event_method"] = event_method
        events["event_coord"] = coord
        events["event_polarity"] = selected_polarity
        events["contact_bodyparts"] = ",".join(contact_bodyparts)
        events["rhythm_bodyparts_aux"] = ",".join(rhythm_bodyparts)
        events["prominence_used_px"] = prominence
        events["expected_period_frames"] = expected_period if expected_period is not None else np.nan
        events["expected_period_s"] = (expected_period / fps) if expected_period is not None and np.isfinite(expected_period) else np.nan

        ordered = [
            "event_id", "frame", "time_s", "support_count", "sources", "source_frames",
            "mean_prominence", "event_signal_value", "event_method", "event_coord", "event_polarity",
            "contact_bodyparts", "rhythm_bodyparts_aux", "prominence_used_px",
            "expected_period_frames", "expected_period_s",
        ]
        events = events[[c for c in ordered if c in events.columns]]

    # Guardar candidatos máximos/mínimos en debug para revisar.
    debug["candidate_max_event"] = 0
    debug.loc[debug.index.intersection(max_events), "candidate_max_event"] = 1
    debug["candidate_min_event"] = 0
    debug.loc[debug.index.intersection(min_events), "candidate_min_event"] = 1
    debug["selected_event"] = 0
    if not events.empty:
        debug.loc[debug.index.intersection(events["frame"].to_numpy(dtype=int)), "selected_event"] = 1

    return events, debug, event_signal, prominence, expected_period, selected_polarity


# =============================================================================
# CONSTRUCCION Y NORMALIZACION DE CICLOS
# =============================================================================

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


# =============================================================================
# GRAFICOS Y EXPORTACION
# =============================================================================

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
        ax.plot(t, debug["distal_x"], linewidth=1.2, label="distal_x toe+foot")
    if "distal_y" in debug.columns:
        ax.plot(t, debug["distal_y"], linewidth=1.0, alpha=0.75, label="distal_y toe+foot")
    if not events.empty:
        ev_t = events["frame"].to_numpy(dtype=int) / float(fps)
        y = debug.loc[events["frame"].to_numpy(dtype=int), "distal_x"].to_numpy(dtype=float) if "distal_x" in debug.columns else debug.loc[events["frame"].to_numpy(dtype=int), "event_signal"].to_numpy(dtype=float)
        ax.scatter(ev_t, y, s=28, marker="o", label="eventos aceptados")
    ax.set_ylabel("posición distal (px)")
    ax.set_title(f"Detección de ciclos: {event_method}, polaridad={selected_polarity}")
    ax.legend(loc="best")

    # 2) señal de evento con candidatos.
    ax = axes[1]
    ax.plot(t, debug["event_signal"], linewidth=1.2, label="event_signal")
    if "candidate_max_event" in debug.columns:
        idx = debug.index[debug["candidate_max_event"] == 1].to_numpy(dtype=int)
        if len(idx):
            ax.scatter(idx / float(fps), debug.loc[idx, "event_signal"], s=18, marker="^", label="candidatos max")
    if "candidate_min_event" in debug.columns:
        idx = debug.index[debug["candidate_min_event"] == 1].to_numpy(dtype=int)
        if len(idx):
            ax.scatter(idx / float(fps), debug.loc[idx, "event_signal"], s=18, marker="v", label="candidatos min")
    if not events.empty:
        idx = events["frame"].to_numpy(dtype=int)
        ax.scatter(idx / float(fps), debug.loc[idx, "event_signal"], s=32, marker="o", label="seleccionados")
    ax.set_ylabel("señal evento")
    ax.legend(loc="best")

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
    smooth_window: int,
    contact_bodyparts: List[str],
    rhythm_bodyparts: List[str],
    event_method: str,
    event_polarity: str,
    selected_polarity: str,
    min_cycle_s: float,
    max_cycle_s: float,
    prominence: float,
    expected_period: Optional[float],
    n_events: int,
    n_cycles: int,
    n_accepted: int,
) -> None:
    with open(params_file, "w", encoding="utf-8") as f:
        f.write("script = 01_preprocesamiento_y_ciclos.py\n")
        f.write("version = distal_x_footstrike_v3\n")
        f.write(f"input_file = {input_file}\n")
        f.write(f"fps = {fps}\n")
        f.write(f"likelihood_min = {likelihood_min}\n")
        f.write(f"max_gap_interpolation = {max_gap}\n")
        f.write(f"smooth_window = {smooth_window}\n")
        f.write(f"contact_bodyparts = {','.join(contact_bodyparts)}\n")
        f.write(f"rhythm_bodyparts_aux = {','.join(rhythm_bodyparts)}\n")
        f.write(f"event_method = {event_method}\n")
        f.write(f"event_polarity_requested = {event_polarity}\n")
        f.write(f"event_polarity_selected = {selected_polarity}\n")
        f.write(f"min_cycle_duration_s = {min_cycle_s}\n")
        f.write(f"max_cycle_duration_s = {max_cycle_s}\n")
        f.write(f"prominence_used_px = {prominence}\n")
        f.write(f"expected_period_frames = {expected_period if expected_period is not None else 'NA'}\n")
        f.write(f"expected_period_s = {(expected_period / fps) if expected_period is not None and np.isfinite(expected_period) else 'NA'}\n")
        f.write(f"n_events = {n_events}\n")
        f.write(f"n_cycles_total = {n_cycles}\n")
        f.write(f"n_cycles_accepted = {n_accepted}\n")
        f.write("cycle_definition = interval between consecutive distal foot-strike-like events from toe+foot distal_x\n")
        f.write("notes = hip/knee/ankle are exported only as auxiliary rhythm signals, not contact detectors.\n")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_pipeline(
    input_file: Path,
    outdir: Path,
    fps: float = FPS,
    likelihood_min: float = LIKELIHOOD_MIN,
    max_gap: int = MAX_GAP_INTERPOLATION,
    smooth_window: int = SMOOTH_WINDOW,
    contact_bodyparts: List[str] = CONTACT_BODY_PARTS,
    rhythm_bodyparts: List[str] = RHYTHM_BODY_PARTS,
    event_method: str = EVENT_METHOD,
    event_polarity: str = EVENT_POLARITY,
    min_cycle_s: float = MIN_CYCLE_DURATION_S,
    max_cycle_s: float = MAX_CYCLE_DURATION_S,
    prominence_px: Optional[float] = DETECTION_PROMINENCE_PX,
    merge_tolerance_frames: int = MERGE_TOLERANCE_FRAMES,
    min_event_support: int = MIN_EVENT_SUPPORT,
    norm_points: int = NORM_POINTS,
) -> Dict[str, Path]:
    input_file = Path(input_file)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_stem(input_file)

    print("\n=== 01_preprocesamiento_y_ciclos.py ===")
    print(f"Entrada: {input_file}")
    print(f"Salida:  {outdir}")
    print(f"FPS:     {fps}")
    print(f"Método:  {event_method} | polaridad solicitada: {event_polarity}")

    raw = read_dlc_file(input_file)
    flat = flatten_dlc_columns(raw)
    clean = clean_coordinates(
        flat=flat,
        likelihood_min=likelihood_min,
        max_gap=max_gap,
        smooth_window=smooth_window,
    )

    events, debug, event_signal, prominence, expected_period, selected_polarity = detect_distal_x_events(
        clean=clean,
        contact_bodyparts=contact_bodyparts,
        rhythm_bodyparts=rhythm_bodyparts,
        fps=fps,
        event_method=event_method,
        event_polarity=event_polarity,
        min_cycle_s=min_cycle_s,
        max_cycle_s=max_cycle_s,
        prominence_px=prominence_px,
        merge_tolerance_frames=merge_tolerance_frames,
        min_event_support=min_event_support,
        smooth_window=smooth_window,
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

    normalized = normalize_cycles(clean=clean, cycles=cycles, n_points=norm_points, accepted_only=True)

    clean_csv = outdir / f"{stem}_clean_coords.csv"
    events_csv = outdir / f"{stem}_events_detected.csv"
    cycles_csv = outdir / f"{stem}_gait_cycles.csv"
    norm_csv = outdir / f"{stem}_normalized_cycles.csv"
    debug_csv = outdir / f"{stem}_cycle_detection_signals.csv"
    check_png = outdir / f"{stem}_cycle_detection_check.png"
    params_txt = outdir / f"{stem}_params.txt"

    clean.to_csv(clean_csv, index=True)
    events.to_csv(events_csv, index=False)
    cycles.to_csv(cycles_csv, index=False)
    normalized.to_csv(norm_csv, index=False)
    debug.to_csv(debug_csv, index=True)

    plot_cycle_detection_check(
        clean=clean,
        events=events,
        cycles=cycles,
        debug=debug,
        out_png=check_png,
        fps=fps,
        event_method=event_method,
        selected_polarity=selected_polarity,
    )

    n_events = 0 if events.empty else len(events)
    n_cycles = 0 if cycles.empty else len(cycles)
    n_accepted = 0 if cycles.empty else int((cycles["accepted"] == 1).sum())

    write_params(
        params_file=params_txt,
        input_file=input_file,
        fps=fps,
        likelihood_min=likelihood_min,
        max_gap=max_gap,
        smooth_window=smooth_window,
        contact_bodyparts=parse_bodypart_list(contact_bodyparts),
        rhythm_bodyparts=parse_bodypart_list(rhythm_bodyparts),
        event_method=event_method,
        event_polarity=event_polarity,
        selected_polarity=selected_polarity,
        min_cycle_s=min_cycle_s,
        max_cycle_s=max_cycle_s,
        prominence=prominence,
        expected_period=expected_period,
        n_events=n_events,
        n_cycles=n_cycles,
        n_accepted=n_accepted,
    )

    print("\nArchivos generados:")
    print(f"  Coordenadas limpias:      {clean_csv}")
    print(f"  Eventos detectados:       {events_csv}")
    print(f"  Ciclos de marcha:         {cycles_csv}")
    print(f"  Ciclos normalizados:      {norm_csv}")
    print(f"  Señales de detección:     {debug_csv}")
    print(f"  Gráfico de control:       {check_png}")
    print(f"  Parámetros:               {params_txt}")
    print("\nResumen:")
    print(f"  Eventos detectados:       {n_events}")
    print(f"  Ciclos totales:           {n_cycles}")
    print(f"  Ciclos aceptados:         {n_accepted}")
    print(f"  Polaridad seleccionada:   {selected_polarity}")
    if expected_period is not None and np.isfinite(expected_period):
        print(f"  Periodo estimado:         {expected_period:.2f} frames = {expected_period/fps:.3f} s")
    print("\nIMPORTANTE: revisar siempre el PNG de control antes de seguir al código 02/03.\n")

    return {
        "clean_coords": clean_csv,
        "events_detected": events_csv,
        "gait_cycles": cycles_csv,
        "normalized_cycles": norm_csv,
        "cycle_detection_signals": debug_csv,
        "cycle_detection_check": check_png,
        "params": params_txt,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Primera parte: leer DLC, limpiar coordenadas, detectar ciclos con toe+foot distal_x y normalizar 0-100%."
    )
    parser.add_argument("input_file", type=str, help="Archivo .h5 o .csv de DeepLabCut.")
    parser.add_argument("--outdir", type=str, default="salida_01_ciclos", help="Carpeta de salida.")
    parser.add_argument("--fps", type=float, default=FPS, help="Frames por segundo del video. Default: 30.")
    parser.add_argument("--likelihood-min", type=float, default=LIKELIHOOD_MIN, help="Umbral mínimo de likelihood.")
    parser.add_argument("--max-gap", type=int, default=MAX_GAP_INTERPOLATION, help="Gaps máximos a interpolar.")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW, help="Ventana de suavizado; se fuerza impar.")
    parser.add_argument("--contact-bodyparts", type=str, default=",".join(CONTACT_BODY_PARTS), help="Puntos distales para detectar ciclos. Ej: toe,foot")
    parser.add_argument("--rhythm-bodyparts", type=str, default=",".join(RHYTHM_BODY_PARTS), help="Puntos auxiliares de ritmo para exportar. No definen contacto.")
    parser.add_argument("--event-method", type=str, default=EVENT_METHOD, choices=["distal_x", "distal_y", "velocity_x"], help="Método de detección.")
    parser.add_argument("--event-polarity", type=str, default=EVENT_POLARITY, choices=["auto", "max", "min"], help="Extremo que define el inicio del ciclo.")
    parser.add_argument("--min-cycle-s", type=float, default=MIN_CYCLE_DURATION_S, help="Duración mínima aceptada del ciclo en segundos.")
    parser.add_argument("--max-cycle-s", type=float, default=MAX_CYCLE_DURATION_S, help="Duración máxima aceptada del ciclo en segundos.")
    parser.add_argument("--prominence", type=float, default=None, help="Prominencia manual en pixeles. Si se omite, se calcula automático.")
    parser.add_argument("--merge-tolerance-frames", type=int, default=MERGE_TOLERANCE_FRAMES, help="Tolerancia para soporte toe/foot.")
    parser.add_argument("--min-event-support", type=int, default=MIN_EVENT_SUPPORT, help="Soporte mínimo de señales. Recomendado 1 o 2.")
    parser.add_argument("--norm-points", type=int, default=NORM_POINTS, help="Puntos para normalización 0-100%. Default: 101.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        run_pipeline(
            input_file=Path(args.input_file),
            outdir=Path(args.outdir),
            fps=args.fps,
            likelihood_min=args.likelihood_min,
            max_gap=args.max_gap,
            smooth_window=args.smooth_window,
            contact_bodyparts=parse_bodypart_list(args.contact_bodyparts),
            rhythm_bodyparts=parse_bodypart_list(args.rhythm_bodyparts),
            event_method=args.event_method,
            event_polarity=args.event_polarity,
            min_cycle_s=args.min_cycle_s,
            max_cycle_s=args.max_cycle_s,
            prominence_px=args.prominence,
            merge_tolerance_frames=args.merge_tolerance_frames,
            min_event_support=args.min_event_support,
            norm_points=args.norm_points,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
