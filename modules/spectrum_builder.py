from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from modules.broadening import apply_line_broadening
from modules.config_loader import SimulationConfig
from modules.physics import UnitContext, compute_line_intensity


@dataclass
class SpectrumBuildResult:
    wavelength_grid_nm: np.ndarray
    spectra_raw: Dict[str, np.ndarray]
    candidate_line_count: int
    used_line_count: int
    skipped_line_count: int
    filtered_by_wavelength_count: int
    used_lines: List[dict]
    skipped_lines: List[dict]


def build_wavelength_grid(config: SimulationConfig) -> np.ndarray:
    return np.arange(
        config.wavelength_min_nm,
        config.wavelength_max_nm + config.delta_lambda_nm * 0.5,
        config.delta_lambda_nm,
        dtype=float,
    )


def _make_skipped_record(row: pd.Series, reason: str) -> dict:
    return {
        "element": row.get("element"),
        "ion_label": row.get("ion_label"),
        "attempted_wavelength_observed_air_nm": row.get("wavelength_observed_air_nm"),
        "attempted_wavelength_ritz_air_nm": row.get("wavelength_ritz_air_nm"),
        "Aki": row.get("Aki"),
        "Ei_cm1": row.get("Ei_cm1"),
        "Ek_cm1": row.get("Ek_cm1"),
        "J_lower": row.get("J_lower_raw"),
        "J_upper": row.get("J_upper_raw"),
        "skipped_reason": reason,
        "source_file": row.get("source_file"),
        "source_sheet": row.get("source_sheet"),
        "source_row_number": row.get("source_row_number"),
    }


def _select_wavelength(row: pd.Series) -> tuple[float | None, str | None]:
    observed = row.get("wavelength_observed_air_nm")
    ritz = row.get("wavelength_ritz_air_nm")
    if observed is not None and not pd.isna(observed):
        return float(observed), "observed"
    if ritz is not None and not pd.isna(ritz):
        return float(ritz), "ritz"
    return None, None


def build_spectra(
    *,
    config: SimulationConfig,
    lines_table: pd.DataFrame,
    partition_functions: Dict[str, Dict[int, float]],
    ion_fractions: Dict[str, Dict[int, float]],
    unit_context: UnitContext,
    logger,
) -> SpectrumBuildResult:
    wavelength_grid_nm = build_wavelength_grid(config)
    if config.delta_lambda_nm > config.fixed_fwhm_nm / 5.0:
        logger.warning(
            "Sampling warning: delta_lambda_nm = %.6f is larger than fixed_fwhm_nm / 5 = %.6f. "
            "Line shapes may be under-sampled.",
            config.delta_lambda_nm,
            config.fixed_fwhm_nm / 5.0,
        )

    spectra_raw: Dict[str, np.ndarray] = {"total": np.zeros_like(wavelength_grid_nm, dtype=float)}
    for target in config.targets:
        spectra_raw[target.raw] = np.zeros_like(wavelength_grid_nm, dtype=float)

    expanded_min = config.wavelength_min_nm - 20.0 * config.fixed_fwhm_nm
    expanded_max = config.wavelength_max_nm + 20.0 * config.fixed_fwhm_nm

    candidate_line_count = 0
    used_lines: List[dict] = []
    skipped_lines: List[dict] = []
    filtered_by_wavelength_count = 0

    for _, row in lines_table.iterrows():
        ion_parse_error = row.get("ion_parse_error")
        if ion_parse_error is not None and not pd.isna(ion_parse_error):
            skipped_lines.append(_make_skipped_record(row, str(ion_parse_error)))
            logger.warning(
                "Skipping line at source row %s because Ion parsing failed: %s",
                row.get("source_row_number"),
                ion_parse_error,
            )
            continue

        wavelength_used_nm, wavelength_source = _select_wavelength(row)
        if wavelength_used_nm is None:
            skipped_lines.append(_make_skipped_record(row, "Observed and Ritz air wavelengths are both missing."))
            logger.warning(
                "Skipping line at source row %s because both Observed and Ritz wavelengths are missing.",
                row.get("source_row_number"),
            )
            continue

        if not (expanded_min <= wavelength_used_nm <= expanded_max):
            filtered_by_wavelength_count += 1
            skipped_lines.append(
                _make_skipped_record(
                    row,
                    "Line center is outside the requested wavelength window after the 20*FWHM margin.",
                )
            )
            continue

        candidate_line_count += 1

        if row.get("Aki") is None or pd.isna(row.get("Aki")) or float(row.get("Aki")) <= 0.0:
            skipped_lines.append(_make_skipped_record(row, "Aki is missing, invalid, or non-positive."))
            continue
        if row.get("Ek_cm1") is None or pd.isna(row.get("Ek_cm1")):
            skipped_lines.append(_make_skipped_record(row, "Ek_cm1 is missing or invalid."))
            continue
        if row.get("J_upper") is None or pd.isna(row.get("J_upper")):
            parse_error = row.get("J_upper_parse_error")
            reason = parse_error if parse_error is not None and not pd.isna(parse_error) else "J_upper is missing or invalid."
            skipped_lines.append(_make_skipped_record(row, str(reason)))
            continue
        if row.get("Ei_cm1") is not None and not pd.isna(row.get("Ei_cm1")):
            if float(row.get("Ek_cm1")) < float(row.get("Ei_cm1")):
                skipped_lines.append(_make_skipped_record(row, "Data anomaly: Ek_cm1 < Ei_cm1."))
                continue

        element = str(row.get("element"))
        ion_stage = int(row.get("ion_stage"))
        partition_function = partition_functions[element][ion_stage]
        ion_fraction = ion_fractions[element][ion_stage]

        line_intensity_raw, g_upper = compute_line_intensity(
            wavelength_nm=wavelength_used_nm,
            Aki=float(row.get("Aki")),
            Ek_cm1=float(row.get("Ek_cm1")),
            J_upper=float(row.get("J_upper")),
            element_mole_fraction=config.element_mole_fractions[element],
            ion_fraction=ion_fraction,
            partition_function=partition_function,
            intensity_mode=config.intensity_mode,
            unit_context=unit_context,
        )

        profile = apply_line_broadening(
            broadening_mode=config.broadening_mode,
            wavelength_grid_nm=wavelength_grid_nm,
            center_wavelength_nm=wavelength_used_nm,
            line_area=line_intensity_raw,
            fixed_fwhm_nm=config.fixed_fwhm_nm,
            line_record=row.to_dict(),
            config=config,
        )

        spectra_raw["total"] += profile
        for target in config.targets:
            if target.kind == "element" and target.element == element:
                spectra_raw[target.raw] += profile
            if target.kind == "ion" and target.element == element and target.ion_stage == ion_stage:
                spectra_raw[target.raw] += profile

        used_lines.append(
            {
                "element": element,
                "ion_stage": ion_stage,
                "ion_label": row.get("ion_label"),
                "element_mole_fraction": config.element_mole_fractions[element],
                "wavelength_used_nm": wavelength_used_nm,
                "wavelength_source": wavelength_source,
                "Aki": float(row.get("Aki")),
                "Ei_cm1": row.get("Ei_cm1"),
                "Ek_cm1": float(row.get("Ek_cm1")),
                "J_lower": row.get("J_lower"),
                "J_upper": float(row.get("J_upper")),
                "g_upper": g_upper,
                "partition_function": partition_function,
                "ion_fraction": ion_fraction,
                "line_intensity_raw": line_intensity_raw,
                "intensity_mode": config.intensity_mode,
                "fwhm_nm_used": config.fixed_fwhm_nm,
                "source_file": row.get("source_file"),
                "source_sheet": row.get("source_sheet"),
                "source_row_number": row.get("source_row_number"),
            }
        )

    return SpectrumBuildResult(
        wavelength_grid_nm=wavelength_grid_nm,
        spectra_raw=spectra_raw,
        candidate_line_count=candidate_line_count,
        used_line_count=len(used_lines),
        skipped_line_count=len(skipped_lines),
        filtered_by_wavelength_count=filtered_by_wavelength_count,
        used_lines=used_lines,
        skipped_lines=skipped_lines,
    )
