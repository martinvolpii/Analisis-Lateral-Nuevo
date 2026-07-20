#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_analisis_WT_vs_SOD1.py

Análisis inferencial P30 Day 1, auditado.

Diseño experimental P30 Day 1
------------------------------
- Velocidad de cinta: 20 cm/s para todos los animales.
- Sexo: machos para todos los animales.
- Camada: todos pertenecen a la MISMA camada.
- WT: 860, 861, 863, 867.
- SOD1: 856, 857, 859, 862, 864, 865.

Consecuencia estadística de una sola camada
-------------------------------------------
Con un único nivel de "litter" NO puede estimarse un efecto aleatorio de camada.
No se añade (1|litter), porque sería no identificable. La comparación de genotipo
es un contraste ENTRE HERMANOS dentro de esta camada. Esto controla sexo, velocidad
y ambiente de camada dentro de la cohorte, pero limita la generalización a otras
camadas. Para inferencia poblacional robusta se requieren varias camadas.

Principios
----------
- Unidad inferencial primaria = ANIMAL, nunca ciclo.
- stride_duration_s usa TODOS los ciclos validados: no depende de toe-off.
- stance/swing/toe clearance respetan accepted_temporal.
- Prueba primaria = permutación exacta bilateral del estadístico t de Welch
  (studentized permutation). Con n=10 y n_WT=4 se enumeran las 210 asignaciones.
- Se reporta permutación de diferencia de medias como sensibilidad.
- Corrección Benjamini-Hochberg FDR sobre las familias declaradas.
- Los tamaños de efecto se reportan aun cuando p/q no sean significativos.
- No se realizan afirmaciones confirmatorias basadas en p nominal aislado.

Entrada
-------
Excel generado por 04_validacion_estadistica_y_excel.py, con hojas:
- animal_means
- cycles_individual_all

Uso
---
python 05_analisis_WT_vs_SOD1.py validacion_estadistica_dlc.xlsx --outdir resultados_grupos
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_WT = {"860", "861", "863", "867"}
DEFAULT_SOD1 = {"856", "857", "859", "862", "864", "865"}

SCALAR_VARIABLES = [
    ("hip_range_deg", "hip_range_deg_mean", "°"),
    ("knee_range_deg", "knee_range_deg_mean", "°"),
    ("ankle_range_deg", "ankle_range_deg_mean", "°"),
    ("foot_range_deg", "foot_range_deg_mean", "°"),
    ("stride_duration_s", "stride_duration_s_mean", "s"),
    ("stance_duration_s", "stance_duration_s_mean", "s"),
    ("swing_duration_s", "swing_duration_s_mean", "s"),
    ("stance_percent", "stance_percent_mean", "%"),
    ("swing_percent", "swing_percent_mean", "%"),
    ("toe_clearance_px", "toe_clearance_px_mean", "px"),
]

VARIABILITY_VARIABLES = [
    ("hip_range_deg", "°"),
    ("knee_range_deg", "°"),
    ("ankle_range_deg", "°"),
    ("foot_range_deg", "°"),
    ("stride_duration_s", "s"),
    ("stance_percent", "%"),
    ("toe_clearance_px", "px"),
]


def parse_ids(text: str) -> set[str]:
    return {x.strip() for x in str(text).split(",") if x.strip()}


def clean_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def welch_t(x: np.ndarray, y: np.ndarray) -> float:
    """Welch t para SOD1 vs WT."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    denom = np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y))
    return float((np.mean(x) - np.mean(y)) / denom) if np.isfinite(denom) and denom > 0 else np.nan


def exact_permutation(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    n_wt: int,
    studentized: bool,
) -> tuple[float, float, int]:
    """Permutación exacta bilateral entre animales."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=str)
    if len(values) != len(labels):
        raise ValueError("values y labels deben tener igual longitud.")

    if studentized:
        observed = welch_t(values[labels == "SOD1"], values[labels == "WT"])
    else:
        observed = float(np.mean(values[labels == "SOD1"]) - np.mean(values[labels == "WT"]))

    permuted: list[float] = []
    for wt_idx in itertools.combinations(range(len(values)), n_wt):
        wt_mask = np.zeros(len(values), dtype=bool)
        wt_mask[list(wt_idx)] = True
        sod_vals = values[~wt_mask]
        wt_vals = values[wt_mask]
        stat = welch_t(sod_vals, wt_vals) if studentized else float(np.mean(sod_vals) - np.mean(wt_vals))
        permuted.append(stat)

    perm = np.asarray(permuted, dtype=float)
    finite = np.isfinite(perm)
    if not np.isfinite(observed) or finite.sum() == 0:
        return observed, np.nan, int(finite.sum())
    p = float(np.mean(np.abs(perm[finite]) >= abs(observed) - 1e-12))
    return observed, p, int(finite.sum())


def hedges_g(x: Iterable[float], y: Iterable[float]) -> float:
    """Hedges g; x=SOD1, y=WT."""
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan
    d = (np.mean(x) - np.mean(y)) / np.sqrt(pooled_var)
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return float(correction * d)


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    x = list(map(float, x))
    y = list(map(float, y))
    if not x or not y:
        return np.nan
    gt = sum(a > b for a in x for b in y)
    lt = sum(a < b for a in x for b in y)
    return float((gt - lt) / (len(x) * len(y)))


def bh_fdr(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(p))[0]
    if len(finite_idx) == 0:
        return q

    pf = p[finite_idx]
    order = np.argsort(pf)
    ranked = pf[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adjusted[i] = running
    inverse = np.empty(m, dtype=int)
    inverse[order] = np.arange(m)
    q[finite_idx] = adjusted[inverse]
    return q


def validate_assignment(ids: set[str], wt: set[str], sod1: set[str]) -> None:
    if wt & sod1:
        raise ValueError(f"IDs presentes en ambos grupos: {sorted(wt & sod1)}")
    expected = wt | sod1
    if ids != expected:
        raise ValueError(
            "Los IDs del Excel no coinciden exactamente con la asignación.\n"
            f"En Excel: {sorted(ids)}\nEsperados: {sorted(expected)}"
        )


def scalar_analysis(animal_means: pd.DataFrame, wt: set[str], sod1: set[str]) -> pd.DataFrame:
    df = animal_means.copy()
    df["animal_id"] = clean_ids(df["animal_id"])
    df["group"] = np.where(df["animal_id"].isin(wt), "WT", "SOD1")
    validate_assignment(set(df["animal_id"]), wt, sod1)
    n_wt = len(wt)

    rows = []
    for variable, column, unit in SCALAR_VARIABLES:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        labels = df["group"].to_numpy()
        finite = np.isfinite(values)
        values, labels = values[finite], labels[finite]
        sod = values[labels == "SOD1"]
        wt_vals = values[labels == "WT"]

        diff = float(np.mean(sod) - np.mean(wt_vals))
        t_stat, p_student, n_perm = exact_permutation(
            values, labels, n_wt=n_wt, studentized=True
        )
        _, p_mean, _ = exact_permutation(
            values, labels, n_wt=n_wt, studentized=False
        )

        rows.append({
            "variable": variable,
            "unit": unit,
            "n_SOD1": len(sod),
            "mean_SOD1": float(np.mean(sod)),
            "sd_SOD1": float(np.std(sod, ddof=1)) if len(sod) > 1 else np.nan,
            "n_WT": len(wt_vals),
            "mean_WT": float(np.mean(wt_vals)),
            "sd_WT": float(np.std(wt_vals, ddof=1)) if len(wt_vals) > 1 else np.nan,
            "difference_SOD1_minus_WT": diff,
            "percent_difference_vs_WT": float(100 * diff / np.mean(wt_vals))
                if np.mean(wt_vals) != 0 else np.nan,
            "hedges_g": hedges_g(sod, wt_vals),
            "cliffs_delta": cliffs_delta(sod, wt_vals),
            "welch_t_observed": t_stat,
            "exact_studentized_permutation_p": p_student,
            "exact_mean_difference_permutation_p_sensitivity": p_mean,
            "n_exact_permutations": n_perm,
        })

    out = pd.DataFrame(rows)
    out["BH_FDR_q_studentized_10_variables"] = bh_fdr(
        out["exact_studentized_permutation_p"].tolist()
    )
    out["BH_FDR_q_mean_difference_sensitivity"] = bh_fdr(
        out["exact_mean_difference_permutation_p_sensitivity"].tolist()
    )
    return out


def variability_analysis(cycles: pd.DataFrame, wt: set[str], sod1: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = cycles.copy()
    df["animal_id"] = clean_ids(df["animal_id"])
    df["group"] = np.where(df["animal_id"].isin(wt), "WT", "SOD1")
    validate_assignment(set(df["animal_id"]), wt, sod1)

    per_animal = []
    for variable, unit in VARIABILITY_VARIABLES:
        if variable not in df.columns:
            continue
        for animal_id, g in df.groupby("animal_id", sort=True):
            vals = pd.to_numeric(g[variable], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) < 2:
                continue
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            cv = float(sd / abs(mean) * 100.0) if mean != 0 else np.nan
            per_animal.append({
                "animal_id": animal_id,
                "group": "WT" if animal_id in wt else "SOD1",
                "variable": variable,
                "unit": unit,
                "n_cycles": len(vals),
                "mean": mean,
                "sd": sd,
                "cv_percent": cv,
            })

    per = pd.DataFrame(per_animal)
    rows = []
    n_wt = len(wt)
    for variable, g in per.groupby("variable", sort=False):
        values = pd.to_numeric(g["cv_percent"], errors="coerce").to_numpy(dtype=float)
        labels = g["group"].to_numpy()
        finite = np.isfinite(values)
        values, labels = values[finite], labels[finite]
        sod = values[labels == "SOD1"]
        wt_vals = values[labels == "WT"]
        diff = float(np.mean(sod) - np.mean(wt_vals))
        t_stat, p_student, n_perm = exact_permutation(
            values, labels, n_wt=n_wt, studentized=True
        )
        _, p_mean, _ = exact_permutation(
            values, labels, n_wt=n_wt, studentized=False
        )
        rows.append({
            "variable": variable,
            "n_SOD1": len(sod),
            "mean_CV_SOD1": float(np.mean(sod)),
            "sd_CV_SOD1": float(np.std(sod, ddof=1)) if len(sod) > 1 else np.nan,
            "n_WT": len(wt_vals),
            "mean_CV_WT": float(np.mean(wt_vals)),
            "sd_CV_WT": float(np.std(wt_vals, ddof=1)) if len(wt_vals) > 1 else np.nan,
            "difference_CV_percentage_points": diff,
            "hedges_g_CV": hedges_g(sod, wt_vals),
            "welch_t_observed_CV": t_stat,
            "exact_studentized_permutation_p": p_student,
            "exact_mean_difference_permutation_p_sensitivity": p_mean,
            "n_exact_permutations": n_perm,
        })

    comp = pd.DataFrame(rows)
    if not comp.empty:
        comp["BH_FDR_q_studentized_variability"] = bh_fdr(
            comp["exact_studentized_permutation_p"].tolist()
        )
        comp["BH_FDR_q_mean_difference_sensitivity"] = bh_fdr(
            comp["exact_mean_difference_permutation_p_sensitivity"].tolist()
        )
    return per, comp


def temporal_qc_by_group(cycles: pd.DataFrame, wt: set[str]) -> pd.DataFrame:
    df = cycles.copy()
    df["animal_id"] = clean_ids(df["animal_id"])
    df["group"] = np.where(df["animal_id"].isin(wt), "WT", "SOD1")

    rows = []
    for group, g in df.groupby("group", sort=True):
        accepted = pd.to_numeric(g.get("accepted_temporal"), errors="coerce").eq(1)
        reasons = (
            g.loc[~accepted, "reject_reason"].fillna("sin_razon").astype(str).value_counts().to_dict()
            if "reject_reason" in g.columns else {}
        )
        rows.append({
            "group": group,
            "n_animals": int(g["animal_id"].nunique()),
            "cycles_total": int(len(g)),
            "stride_valid": int(pd.to_numeric(g.get("stride_duration_s"), errors="coerce").notna().sum()),
            "temporal_phase_valid": int(accepted.sum()),
            "temporal_phase_invalid": int((~accepted).sum()),
            "temporal_phase_valid_percent": float(100 * accepted.mean()) if len(g) else np.nan,
            "invalid_reasons": json.dumps(reasons, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Análisis auditado WT vs SOD1 para P30 Day 1.")
    ap.add_argument("validation_xlsx", type=Path, help="Excel generado por script 04 corregido.")
    ap.add_argument("--outdir", type=Path, default=Path("resultados_WT_vs_SOD1"))
    ap.add_argument("--wt-ids", default="860,861,863,867")
    ap.add_argument("--sod1-ids", default="856,857,859,862,864,865")
    ap.add_argument("--treadmill-speed-cm-s", type=float, default=20.0)
    ap.add_argument("--sex", type=str, default="male")
    ap.add_argument("--litter-id", type=str, default="P30_single_shared_litter")
    args = ap.parse_args()

    wt = parse_ids(args.wt_ids)
    sod1 = parse_ids(args.sod1_ids)
    if not args.validation_xlsx.exists():
        raise FileNotFoundError(args.validation_xlsx)

    animal_means = pd.read_excel(args.validation_xlsx, sheet_name="animal_means")
    cycles = pd.read_excel(args.validation_xlsx, sheet_name="cycles_individual_all")

    scalar = scalar_analysis(animal_means, wt, sod1)
    per_animal_var, variability = variability_analysis(cycles, wt, sod1)
    qc = temporal_qc_by_group(cycles, wt)

    args.outdir.mkdir(parents=True, exist_ok=True)
    scalar.to_csv(args.outdir / "comparacion_variables_WT_vs_SOD1.csv", index=False)
    per_animal_var.to_csv(args.outdir / "variabilidad_intraanimal_por_animal.csv", index=False)
    variability.to_csv(args.outdir / "comparacion_variabilidad_WT_vs_SOD1.csv", index=False)
    qc.to_csv(args.outdir / "QC_temporal_por_grupo.csv", index=False)

    metadata = {
        "treadmill_speed_cm_s": args.treadmill_speed_cm_s,
        "sex": args.sex,
        "litter_id": args.litter_id,
        "n_litters": 1,
        "WT_ids": sorted(wt),
        "SOD1_ids": sorted(sod1),
        "statistical_unit": "animal",
        "primary_test": "exact two-sided studentized permutation (Welch t)",
        "litter_effect_modelled": False,
        "litter_reason": (
            "Only one litter level exists. A litter random effect is not identifiable. "
            "Inference is conditional on this shared litter; replicate independent litters "
            "are required for broad population generalization."
        ),
        "sex_covariate_modelled": False,
        "sex_reason": "All animals are male; sex is controlled by design but cannot be estimated.",
        "speed_covariate_modelled": False,
        "speed_reason": "All animals were tested at the same treadmill speed (20 cm/s).",
    }
    (args.outdir / "study_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== P30 WT vs SOD1 — ANALISIS AUDITADO ===")
    print(f"WT: {sorted(wt)}")
    print(f"SOD1: {sorted(sod1)}")
    print(f"Velocidad: {args.treadmill_speed_cm_s} cm/s | Sexo: {args.sex} | Camada: {args.litter_id}")
    print(
        "ADVERTENCIA DE GENERALIZACION: hay una sola camada; no puede estimarse "
        "un efecto aleatorio de litter. Las conclusiones son exploratorias y "
        "condicionadas a esta camada."
    )
    print("\nVariables escalares:")
    print(scalar.to_string(index=False))
    print("\nVariabilidad intraanimal:")
    print(variability.to_string(index=False))


if __name__ == "__main__":
    main()
