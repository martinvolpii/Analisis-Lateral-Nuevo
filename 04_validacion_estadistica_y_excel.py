#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_validacion_estadistica_y_excel.py

Cuarta parte del pipeline para análisis cinemático lateral con DeepLabCut.

Este script NO vuelve a procesar coordenadas ni recalcula ciclos. Usa las salidas de:
    02_goniometria_lateral_por_ciclos.py
    03_variables_temporales_y_toe_clearance.py

Objetivo:
    Generar un Excel final ordenado para validación del pipeline, descriptivos,
    datos por ciclo, datos balanceados, promedios por animal y normalidad.

Entradas esperadas en una carpeta raíz, buscadas recursivamente:
    *_cycle_angle_ranges.csv
    *_gait_temporal_by_cycle.csv

Salida:
    validacion_estadistica_dlc.xlsx

Uso básico:
    python 04_validacion_estadistica_y_excel.py \
        --input-dir dlc_pipeline_run_20260703 \
        --out validacion_estadistica_dlc.xlsx \
        --include 856,857,859,860 \
        --exclude 861,862,863 \
        --balanced-n 10

Uso con análisis de sensibilidad incluyendo R6:
    python 04_validacion_estadistica_y_excel.py \
        --input-dir dlc_pipeline_run_20260703 \
        --out validacion_estadistica_dlc.xlsx \
        --include 856,857,859,860 \
        --exclude 861,862,863 \
        --sensitivity-include 856,857,859,860,861 \
        --balanced-n 10

Notas metodológicas:
    - La unidad estadística principal es el animal, no el ciclo.
    - Los ciclos individuales se conservan como descriptivos y control de dispersión.
    - La hoja 'animal_means' debe usarse para inferencia grupal posterior.
    - La hoja 'balanced_cycles' usa hasta N ciclos por animal para visualización
      balanceada, evitando que un animal con muchos ciclos pese más visualmente.

Autor: pipeline preparado para análisis DLC de marcha murina.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import shapiro
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================================================================
# VARIABLES DEL PIPELINE
# =============================================================================

ANGLE_VARIABLES = [
    "hip_range_deg",
    "knee_range_deg",
    "ankle_range_deg",
    "foot_range_deg",
]

TEMPORAL_VARIABLES = [
    "stride_duration_s",
    "stance_duration_s",
    "swing_duration_s",
    "stance_percent",
    "swing_percent",
    "toe_clearance_px",
]

DEFAULT_VARIABLES = ANGLE_VARIABLES + TEMPORAL_VARIABLES

ID_COLUMNS = [
    "animal_id",
    "dataset_id",
    "source_stem",
    "cycle_id",
    "start_frame",
    "end_frame",
]


# =============================================================================
# UTILIDADES
# =============================================================================

def parse_list(value: Optional[str]) -> List[str]:
    """Convierte una lista CSV de IDs (p. ej. '856,857,859' o 'R1,R2') en una lista limpia."""
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_animal_id(text: str) -> str:
    """
    Extrae un ID de animal desde nombres de dataset/archivo.

    Formatos soportados:
      - R1, R2, R10 (formato histórico).
      - 856_P30, 857_P30DLC_..., etc. (formato actual).
      - Un ID numérico al inicio seguido de '_' o '-' como respaldo.

    Para nombres actuales como ``856_P30DLC_...`` devuelve ``856``; P30 se
    conserva dentro de ``dataset_id`` pero no forma parte del identificador único
    del animal.
    """
    text = str(text)

    # Formato histórico: R seguido de dígitos.
    m = re.search(r"(^|[^A-Za-z0-9])(R\d+)([^A-Za-z0-9]|$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"R\d+", text, flags=re.IGNORECASE)
    if m:
        token = m.group(2) if len(m.groups()) >= 2 and m.group(2) else m.group(0)
        return token.upper()

    # Formato actual: 856_P30..., 857_P30..., etc.
    m = re.search(r"(?:^|[^0-9])(\d+)_P\d+", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # Respaldo conservador: ID numérico al inicio antes de '_' o '-'.
    m = re.match(r"^(\d{2,})(?=[_-])", text)
    if m:
        return m.group(1)

    return "UNKNOWN"


def normalize_animal_filter_id(value: str) -> str:
    """Normaliza IDs usados en --include/--exclude al mismo formato de animal_id."""
    token = str(value).strip()
    if not token:
        return token
    parsed = parse_animal_id(token)
    if parsed != "UNKNOWN":
        return parsed.upper()
    if re.fullmatch(r"\d+", token):
        return token
    return token.upper()


def clean_stem_from_suffix(path: Path, suffix: str) -> str:
    """Quita el sufijo estándar del stem de un archivo de salida."""
    name = path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def sem(values: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(arr)
    if n <= 1:
        return np.nan
    return float(np.std(arr, ddof=1) / math.sqrt(n))


def safe_cv_percent(values: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return np.nan
    mean = float(np.mean(arr))
    if mean == 0 or not np.isfinite(mean):
        return np.nan
    return float(np.std(arr, ddof=1) / abs(mean) * 100.0)


def iqr(values: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan
    return float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))


def descriptive_stats(values: Sequence[float]) -> Dict[str, float]:
    """Estadísticos descriptivos estándar para una variable."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(arr))
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "iqr": np.nan,
            "cv_percent": np.nan,
        }
    sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "sd": sd,
        "sem": float(sd / math.sqrt(n)) if n > 1 else np.nan,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "iqr": iqr(arr),
        "cv_percent": safe_cv_percent(arr),
    }


def normality_shapiro(values: Sequence[float]) -> Dict[str, object]:
    """
    Prueba Shapiro-Wilk.

    Requisitos prácticos:
        - mínimo 3 datos finitos
        - scipy instalado
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(arr))
    if not SCIPY_AVAILABLE:
        return {"n": n, "test": "Shapiro-Wilk", "statistic": np.nan, "p_value": np.nan, "normal_p_ge_0_05": np.nan, "note": "scipy_no_disponible"}
    if n < 3:
        return {"n": n, "test": "Shapiro-Wilk", "statistic": np.nan, "p_value": np.nan, "normal_p_ge_0_05": np.nan, "note": "n_menor_3"}
    if np.nanstd(arr) == 0:
        return {"n": n, "test": "Shapiro-Wilk", "statistic": np.nan, "p_value": np.nan, "normal_p_ge_0_05": np.nan, "note": "sin_variabilidad"}
    stat, p = shapiro(arr)
    return {
        "n": n,
        "test": "Shapiro-Wilk",
        "statistic": float(stat),
        "p_value": float(p),
        "normal_p_ge_0_05": bool(p >= 0.05),
        "note": "",
    }


# =============================================================================
# LECTURA DE ARCHIVOS 02/03
# =============================================================================

def find_pipeline_outputs(input_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Busca archivos de rangos angulares y variables temporales recursivamente."""
    input_dir = Path(input_dir)
    angle_files: Dict[str, Path] = {}
    temporal_files: Dict[str, Path] = {}

    for p in sorted(input_dir.rglob("*_cycle_angle_ranges.csv")):
        stem = clean_stem_from_suffix(p, "_cycle_angle_ranges.csv")
        angle_files[stem] = p

    for p in sorted(input_dir.rglob("*_gait_temporal_by_cycle.csv")):
        stem = clean_stem_from_suffix(p, "_gait_temporal_by_cycle.csv")
        temporal_files[stem] = p

    return angle_files, temporal_files


def read_one_dataset(stem: str, angle_path: Path, temporal_path: Optional[Path]) -> pd.DataFrame:
    """Lee y fusiona un dataset por ciclo."""
    angles = pd.read_csv(angle_path)
    animal_id = parse_animal_id(stem)

    # Mantener solo columnas útiles de ángulos y metadatos.
    angle_keep = [c for c in ["cycle_id", "start_frame", "end_frame", "duration_frames", "duration_s"] if c in angles.columns]
    angle_keep += [c for c in ANGLE_VARIABLES if c in angles.columns]
    angles = angles[angle_keep].copy()

    merged = angles.copy()

    if temporal_path is not None and temporal_path.exists():
        temporal = pd.read_csv(temporal_path)
        # Usar solo ciclos temporales aceptados para variables temporales.
        if "accepted_temporal" in temporal.columns:
            temporal_valid = temporal[temporal["accepted_temporal"] == 1].copy()
        else:
            temporal_valid = temporal.copy()

        temporal_keep = [c for c in ["cycle_id", "start_frame", "end_frame", "accepted_temporal", "reject_reason"] if c in temporal_valid.columns]
        temporal_keep += [c for c in TEMPORAL_VARIABLES if c in temporal_valid.columns]
        temporal_valid = temporal_valid[temporal_keep].copy()

        merge_cols = [c for c in ["cycle_id", "start_frame", "end_frame"] if c in merged.columns and c in temporal_valid.columns]
        if not merge_cols:
            merge_cols = ["cycle_id"] if "cycle_id" in merged.columns and "cycle_id" in temporal_valid.columns else []

        if merge_cols:
            merged = pd.merge(merged, temporal_valid, on=merge_cols, how="left", suffixes=("", "_temporal"))
        else:
            # Si no hay llave clara, conservar solo ángulos y advertir con columnas vacías.
            for c in TEMPORAL_VARIABLES:
                if c not in merged.columns:
                    merged[c] = np.nan
            merged["accepted_temporal"] = np.nan
            merged["reject_reason"] = "sin_llave_para_fusion"
    else:
        for c in TEMPORAL_VARIABLES:
            merged[c] = np.nan
        merged["accepted_temporal"] = np.nan
        merged["reject_reason"] = "sin_archivo_temporal"

    merged.insert(0, "animal_id", animal_id)
    merged.insert(1, "dataset_id", stem)
    merged.insert(2, "source_stem", stem)
    merged["angle_file"] = str(angle_path)
    merged["temporal_file"] = str(temporal_path) if temporal_path else ""

    return merged


def load_all_datasets(input_dir: Path) -> pd.DataFrame:
    """Carga todos los pares encontrados de outputs 02/03."""
    angle_files, temporal_files = find_pipeline_outputs(input_dir)
    if not angle_files:
        raise FileNotFoundError(f"No encontré archivos *_cycle_angle_ranges.csv en {input_dir}")

    rows = []
    for stem, angle_path in angle_files.items():
        temporal_path = temporal_files.get(stem)
        rows.append(read_one_dataset(stem, angle_path, temporal_path))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Nunca deduplicar silenciosamente datasets cuyo animal no pudo identificarse.
    # Eso podría colapsar varios animales distintos bajo el ID "UNKNOWN".
    if not out.empty and "animal_id" in out.columns:
        unknown = sorted(out.loc[out["animal_id"].astype(str).eq("UNKNOWN"), "dataset_id"].astype(str).unique())
        if unknown:
            preview = ", ".join(unknown[:10])
            more = " ..." if len(unknown) > 10 else ""
            raise ValueError(
                "No pude identificar el animal en uno o más datasets: "
                f"{preview}{more}. Use nombres tipo 856_P30..., 857_P30... o R1/R2. "
                "Se detuvo el análisis para evitar deduplicar animales distintos como UNKNOWN."
            )
    return out


# =============================================================================
# SELECCION, BALANCEO Y RESUMEN
# =============================================================================

def dataset_quality_table(cycles: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas simples para elegir el mejor dataset si hay duplicados por animal."""
    rows = []
    for dataset_id, g in cycles.groupby("dataset_id", dropna=False):
        animal = g["animal_id"].iloc[0] if "animal_id" in g.columns and len(g) else "UNKNOWN"
        angle_valid_counts = []
        for var in ANGLE_VARIABLES:
            if var in g.columns:
                angle_valid_counts.append(int(pd.to_numeric(g[var], errors="coerce").notna().sum()))
        temporal_valid_counts = []
        for var in TEMPORAL_VARIABLES:
            if var in g.columns:
                temporal_valid_counts.append(int(pd.to_numeric(g[var], errors="coerce").notna().sum()))
        n_angle_valid_min = min(angle_valid_counts) if angle_valid_counts else 0
        n_temporal_valid_min = min(temporal_valid_counts) if temporal_valid_counts else 0
        n_cycles_total = int(len(g))
        n_temporal_accepted = int(pd.to_numeric(g.get("accepted_temporal", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1).sum()) if "accepted_temporal" in g.columns else n_temporal_valid_min
        rows.append({
            "animal_id": animal,
            "dataset_id": dataset_id,
            "n_cycles_total": n_cycles_total,
            "n_angle_valid_min": n_angle_valid_min,
            "n_temporal_valid_min": n_temporal_valid_min,
            "n_temporal_accepted": n_temporal_accepted,
        })
    q = pd.DataFrame(rows)
    if not q.empty:
        q["quality_score"] = (
            q["n_temporal_valid_min"].fillna(0) * 10
            + q["n_temporal_accepted"].fillna(0) * 5
            + q["n_angle_valid_min"].fillna(0)
            + q["n_cycles_total"].fillna(0) * 0.1
        )
    return q.sort_values(["animal_id", "quality_score"], ascending=[True, False]).reset_index(drop=True)


def deduplicate_best_dataset(cycles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Si hay más de un dataset por animal, conserva el de mejor calidad.
    Devuelve ciclos filtrados y tabla de decisión.
    """
    q = dataset_quality_table(cycles)
    if q.empty:
        return cycles, q

    q["selected_dataset"] = False
    selected_datasets = []
    for animal, g in q.groupby("animal_id", sort=True):
        best = g.sort_values(["quality_score", "n_temporal_valid_min", "n_angle_valid_min"], ascending=False).iloc[0]
        selected_datasets.append(best["dataset_id"])
        q.loc[q["dataset_id"] == best["dataset_id"], "selected_dataset"] = True

    filtered = cycles[cycles["dataset_id"].isin(selected_datasets)].copy()
    return filtered.reset_index(drop=True), q


def filter_animals(cycles: pd.DataFrame, include: List[str], exclude: List[str]) -> pd.DataFrame:
    """Aplica criterios de inclusión/exclusión por animal con IDs normalizados."""
    out = cycles.copy()
    animal_ids = out["animal_id"].astype(str).str.upper()
    if include:
        include_set = {normalize_animal_filter_id(x) for x in include if str(x).strip()}
        out = out[animal_ids.isin(include_set)].copy()
        animal_ids = out["animal_id"].astype(str).str.upper()
    if exclude:
        exclude_set = {normalize_animal_filter_id(x) for x in exclude if str(x).strip()}
        out = out[~animal_ids.isin(exclude_set)].copy()
    return out.reset_index(drop=True)


def make_balanced_cycles(cycles: pd.DataFrame, variables: List[str], n_per_animal: int) -> pd.DataFrame:
    """
    Toma hasta n_per_animal ciclos por animal.
    Selección determinística: ciclos ordenados por cycle_id/start_frame.
    """
    if n_per_animal <= 0:
        return pd.DataFrame()
    sort_cols = [c for c in ["animal_id", "cycle_id", "start_frame"] if c in cycles.columns]
    out = cycles.sort_values(sort_cols).copy() if sort_cols else cycles.copy()
    rows = []
    for animal, g in out.groupby("animal_id", sort=True):
        # Para balancear, exigir que al menos una variable principal tenga dato válido.
        valid_mask = pd.Series(False, index=g.index)
        for v in variables:
            if v in g.columns:
                valid_mask = valid_mask | pd.to_numeric(g[v], errors="coerce").notna()
        selected = g.loc[valid_mask].head(n_per_animal).copy()
        selected["balanced_cycle_rank"] = np.arange(1, len(selected) + 1)
        selected["balanced_n_target"] = int(n_per_animal)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def animal_variable_long(cycles: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Convierte ciclos a formato largo: una fila por animal/ciclo/variable."""
    id_cols = [c for c in ID_COLUMNS if c in cycles.columns]
    keep_vars = [v for v in variables if v in cycles.columns]
    if not keep_vars:
        return pd.DataFrame()
    long = cycles[id_cols + keep_vars].melt(
        id_vars=id_cols,
        value_vars=keep_vars,
        var_name="variable",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    return long


def stats_by_animal(cycles: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Descriptivos por animal y variable usando ciclos individuales."""
    rows = []
    for animal, g in cycles.groupby("animal_id", sort=True):
        for var in variables:
            if var not in g.columns:
                continue
            stats = descriptive_stats(g[var])
            stats.update({"animal_id": animal, "variable": var})
            rows.append(stats)
    cols = ["animal_id", "variable", "n", "mean", "sd", "sem", "median", "min", "max", "range", "iqr", "cv_percent"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def animal_means_table(cycles: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Una fila por animal; cada variable es la media de sus ciclos."""
    rows = []
    for animal, g in cycles.groupby("animal_id", sort=True):
        row: Dict[str, object] = {"animal_id": animal}
        row["dataset_id"] = ";".join(sorted(map(str, g["dataset_id"].dropna().unique()))) if "dataset_id" in g.columns else ""
        for var in variables:
            if var in g.columns:
                vals = pd.to_numeric(g[var], errors="coerce").dropna()
                row[f"{var}_n_cycles"] = int(len(vals))
                row[f"{var}_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{var}_sd_intra_animal"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
                row[f"{var}_sem_intra_animal"] = sem(vals)
            else:
                row[f"{var}_n_cycles"] = 0
                row[f"{var}_mean"] = np.nan
                row[f"{var}_sd_intra_animal"] = np.nan
                row[f"{var}_sem_intra_animal"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def general_stats_from_animal_means(animal_means: pd.DataFrame, variables: List[str], label: str) -> pd.DataFrame:
    """Resumen general usando n = animal, no n = ciclo."""
    rows = []
    for var in variables:
        col = f"{var}_mean"
        if col not in animal_means.columns:
            continue
        stats = descriptive_stats(animal_means[col])
        norm = normality_shapiro(animal_means[col])
        row = {"analysis_set": label, "unit": "animal", "variable": var}
        row.update(stats)
        row.update({
            "shapiro_n": norm["n"],
            "shapiro_statistic": norm["statistic"],
            "shapiro_p_value": norm["p_value"],
            "normal_p_ge_0_05": norm["normal_p_ge_0_05"],
            "normality_note": norm["note"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def normality_by_cycles(cycles: pd.DataFrame, variables: List[str], label: str) -> pd.DataFrame:
    """Normalidad usando todos los ciclos. Es descriptiva, no inferencia principal."""
    rows = []
    for var in variables:
        if var not in cycles.columns:
            continue
        norm = normality_shapiro(cycles[var])
        row = {"analysis_set": label, "unit": "cycle_descriptive", "variable": var}
        row.update(norm)
        rows.append(row)
    return pd.DataFrame(rows)


def inclusion_table(all_cycles: pd.DataFrame, selected_cycles: pd.DataFrame, include: List[str], exclude: List[str], sensitivity_include: List[str]) -> pd.DataFrame:
    """Tabla transparente de inclusión/exclusión."""
    all_animals = sorted(all_cycles["animal_id"].dropna().unique().tolist()) if not all_cycles.empty else []
    selected_animals = set(selected_cycles["animal_id"].dropna().unique().tolist()) if not selected_cycles.empty else set()
    rows = []
    for animal in all_animals:
        status = "incluido_principal" if animal in selected_animals else "excluido"
        reason = ""
        if include and animal not in {x.upper() for x in include}:
            reason = "no_en_lista_include"
        if animal in {x.upper() for x in exclude}:
            reason = "en_lista_exclude"
        if sensitivity_include and animal in {x.upper() for x in sensitivity_include} and animal not in selected_animals:
            status = "solo_sensibilidad"
        rows.append({"animal_id": animal, "status": status, "reason": reason})
    return pd.DataFrame(rows)


# =============================================================================
# EXPORTACION A EXCEL
# =============================================================================

def autosize_excel_columns(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame, max_width: int = 42) -> None:
    """Ajuste básico de ancho de columnas usando xlsxwriter."""
    worksheet = writer.sheets[sheet_name]
    for idx, col in enumerate(df.columns):
        sample = df[col].astype(str).replace("nan", "").head(100).tolist()
        width = max([len(str(col))] + [len(x) for x in sample]) + 2
        width = min(max(width, 10), max_width)
        worksheet.set_column(idx, idx, width)


def write_excel(
    out_path: Path,
    sheets: Dict[str, pd.DataFrame],
    metadata: Dict[str, object],
) -> None:
    """Escribe un Excel legible con múltiples hojas."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_fmt = workbook.add_format({"bold": True, "font_size": 14, "bg_color": "#D9EAF7", "border": 1})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#EAF3F8", "border": 1})
        note_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        num_fmt = workbook.add_format({"num_format": "0.000"})

        # Hoja README/metadata.
        meta_rows = [{"campo": k, "valor": v} for k, v in metadata.items()]
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_excel(writer, sheet_name="README", index=False, startrow=3)
        ws = writer.sheets["README"]
        ws.write("A1", "Validación estadística del pipeline DLC", title_fmt)
        ws.write("A2", "La unidad estadística principal es el animal. Los ciclos individuales son descriptivos.", note_fmt)
        ws.set_column("A:A", 32)
        ws.set_column("B:B", 90)
        ws.freeze_panes(4, 0)

        # Hojas de datos.
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            if df is None or df.empty:
                pd.DataFrame({"mensaje": ["Sin datos para esta hoja"]}).to_excel(writer, sheet_name=safe_name, index=False)
                autosize_excel_columns(writer, safe_name, pd.DataFrame({"mensaje": ["Sin datos para esta hoja"]}))
                continue
            df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.sheets[safe_name]
            # Estilo de encabezado.
            for col_idx, col in enumerate(df.columns):
                ws.write(0, col_idx, col, header_fmt)
            autosize_excel_columns(writer, safe_name, df)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(df), max(0, len(df.columns) - 1))
            # Formato numérico general para columnas numéricas.
            for col_idx, col in enumerate(df.columns):
                if pd.api.types.is_numeric_dtype(df[col]):
                    ws.set_column(col_idx, col_idx, None, num_fmt)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def build_validation_workbook(
    input_dir: Path,
    out_path: Path,
    include: List[str],
    exclude: List[str],
    sensitivity_include: List[str],
    balanced_n: int,
    deduplicate: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Ejecuta todo el análisis del script 04 y exporta Excel."""
    all_cycles_raw = load_all_datasets(input_dir)
    variables = [v for v in DEFAULT_VARIABLES if v in all_cycles_raw.columns]

    if deduplicate:
        all_cycles, dataset_selection = deduplicate_best_dataset(all_cycles_raw)
    else:
        all_cycles = all_cycles_raw.copy()
        dataset_selection = dataset_quality_table(all_cycles_raw)
        dataset_selection["selected_dataset"] = True

    main_cycles = filter_animals(all_cycles, include=include, exclude=exclude)
    balanced_cycles = make_balanced_cycles(main_cycles, variables=variables, n_per_animal=balanced_n)
    cycle_long = animal_variable_long(main_cycles, variables=variables)
    stats_animal = stats_by_animal(main_cycles, variables=variables)
    animal_means = animal_means_table(main_cycles, variables=variables)
    general_main = general_stats_from_animal_means(animal_means, variables=variables, label="principal")
    normality_cycles_main = normality_by_cycles(main_cycles, variables=variables, label="principal")

    # Análisis de sensibilidad opcional.
    if sensitivity_include:
        sensitivity_cycles = filter_animals(all_cycles, include=sensitivity_include, exclude=[])
        sensitivity_means = animal_means_table(sensitivity_cycles, variables=variables)
        general_sensitivity = general_stats_from_animal_means(sensitivity_means, variables=variables, label="sensibilidad")
        normality_cycles_sensitivity = normality_by_cycles(sensitivity_cycles, variables=variables, label="sensibilidad")
    else:
        sensitivity_cycles = pd.DataFrame()
        sensitivity_means = pd.DataFrame()
        general_sensitivity = pd.DataFrame()
        normality_cycles_sensitivity = pd.DataFrame()

    general_stats = pd.concat([general_main, general_sensitivity], ignore_index=True)
    normality_all = pd.concat([normality_cycles_main, normality_cycles_sensitivity], ignore_index=True)
    inclusion = inclusion_table(all_cycles, main_cycles, include=include, exclude=exclude, sensitivity_include=sensitivity_include)

    # Reordenar datos por ciclo para legibilidad.
    preferred_cols = [c for c in ID_COLUMNS if c in main_cycles.columns] + variables
    other_cols = [c for c in main_cycles.columns if c not in preferred_cols]
    cycles_sheet = main_cycles[preferred_cols + other_cols].copy() if not main_cycles.empty else main_cycles
    balanced_sheet = balanced_cycles[[c for c in preferred_cols + ["balanced_cycle_rank", "balanced_n_target"] if c in balanced_cycles.columns] + [c for c in balanced_cycles.columns if c not in preferred_cols + ["balanced_cycle_rank", "balanced_n_target"]]].copy() if not balanced_cycles.empty else balanced_cycles

    sheets = {
        "dataset_selection": dataset_selection,
        "inclusion_exclusion": inclusion,
        "cycles_individual": cycles_sheet,
        "cycles_long": cycle_long,
        "balanced_cycles": balanced_sheet,
        "stats_by_animal": stats_animal,
        "animal_means": animal_means,
        "general_stats_n_animal": general_stats,
        "normality_cycles_desc": normality_all,
        "sensitivity_cycles": sensitivity_cycles,
        "sensitivity_animal_means": sensitivity_means,
    }

    metadata = {
        "input_dir": str(Path(input_dir).resolve()),
        "output_file": str(Path(out_path).resolve()),
        "include_principal": ",".join(include) if include else "todos_menos_exclude",
        "exclude_principal": ",".join(exclude) if exclude else "ninguno",
        "sensitivity_include": ",".join(sensitivity_include) if sensitivity_include else "no_aplica",
        "balanced_n_per_animal": balanced_n,
        "deduplicate_best_dataset_per_animal": deduplicate,
        "variables": ", ".join(variables),
        "statistical_unit_main": "animal",
        "note": "Usar animal_means y general_stats_n_animal para inferencia. cycles_individual es descriptivo/control de dispersión.",
    }

    write_excel(out_path=out_path, sheets=sheets, metadata=metadata)
    return sheets


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera Excel de validación estadística y normalidad desde outputs 02/03 del pipeline DLC."
    )
    p.add_argument("--input-dir", required=True, help="Carpeta raíz donde están las salidas del pipeline 02 y 03.")
    p.add_argument("--out", default="validacion_estadistica_dlc.xlsx", help="Ruta del Excel de salida.")
    p.add_argument("--include", default="", help="Animales a incluir en análisis principal, separados por coma. Ej: 856,857,859,860 (también admite R1,R2,...)")
    p.add_argument("--exclude", default="", help="Animales a excluir del análisis principal, separados por coma. Ej: 861,862,863 (también admite R3,R6,...)")
    p.add_argument("--sensitivity-include", default="", help="Animales para análisis de sensibilidad. Ej: 856,857,859,860,861")
    p.add_argument("--balanced-n", type=int, default=10, help="Máximo de ciclos por animal para hoja balanceada.")
    p.add_argument("--no-deduplicate", action="store_true", help="No elegir automáticamente el mejor dataset si hay duplicados por animal.")
    return p


def main() -> None:
    args = make_argparser().parse_args()
    include = parse_list(args.include)
    exclude = parse_list(args.exclude)
    sensitivity_include = parse_list(args.sensitivity_include)

    sheets = build_validation_workbook(
        input_dir=Path(args.input_dir),
        out_path=Path(args.out),
        include=include,
        exclude=exclude,
        sensitivity_include=sensitivity_include,
        balanced_n=int(args.balanced_n),
        deduplicate=not bool(args.no_deduplicate),
    )

    print("Excel generado correctamente:", args.out)
    print("Hojas generadas:", ", ".join(sheets.keys()))


if __name__ == "__main__":
    main()
