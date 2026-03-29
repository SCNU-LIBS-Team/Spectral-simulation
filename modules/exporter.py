from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from modules.config_loader import SimulationConfig
from modules.parsers import int_to_roman
from modules.spectrum_builder import SpectrumBuildResult


def export_cleaned_dataframe(
    *,
    dataframe: pd.DataFrame,
    output_path: Path,
    sheet_name: str,
    logger,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info("Exported cleaned %s table: %s", sheet_name, output_path)
    return output_path


def _build_spectrum_dataframe(
    wavelength_grid_nm: np.ndarray,
    raw_values: np.ndarray,
    total_max_raw: float,
) -> pd.DataFrame:
    if total_max_raw > 0.0:
        norm_values = raw_values / total_max_raw
    else:
        norm_values = np.zeros_like(raw_values)

    return pd.DataFrame(
        {
            "wavelength_nm": wavelength_grid_nm,
            "raw": raw_values,
            "norm": norm_values,
        }
    )


def _target_filename(target) -> str:
    if target.kind == "element":
        return f"continuous_{target.element}.xlsx"
    return f"continuous_{target.element}_{int_to_roman(int(target.ion_stage))}.xlsx"


def export_continuous_spectra(
    *,
    build_result: SpectrumBuildResult,
    config: SimulationConfig,
    output_dir: Path,
    logger,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_max_raw = float(np.max(build_result.spectra_raw["total"])) if build_result.spectra_raw["total"].size else 0.0
    if total_max_raw <= 0.0:
        logger.warning("Total spectrum max(raw) is not positive. All normalized outputs will be zero.")

    exported_files: List[Path] = []
    total_path = output_dir / "continuous_total.xlsx"
    total_df = _build_spectrum_dataframe(
        wavelength_grid_nm=build_result.wavelength_grid_nm,
        raw_values=build_result.spectra_raw["total"],
        total_max_raw=total_max_raw,
    )
    with pd.ExcelWriter(total_path, engine="openpyxl") as writer:
        total_df.to_excel(writer, sheet_name="spectrum", index=False)
    exported_files.append(total_path)

    for target in config.targets:
        target_path = output_dir / _target_filename(target)
        target_df = _build_spectrum_dataframe(
            wavelength_grid_nm=build_result.wavelength_grid_nm,
            raw_values=build_result.spectra_raw[target.raw],
            total_max_raw=total_max_raw,
        )
        with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
            target_df.to_excel(writer, sheet_name="spectrum", index=False)
        exported_files.append(target_path)

    return exported_files


def export_discrete_line_tables(
    *,
    build_result: SpectrumBuildResult,
    output_dir: Path,
    logger,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    used_path = output_dir / "discrete_lines_used.xlsx"
    skipped_path = output_dir / "discrete_lines_skipped.xlsx"

    used_df = pd.DataFrame(build_result.used_lines)
    skipped_df = pd.DataFrame(build_result.skipped_lines)

    with pd.ExcelWriter(used_path, engine="openpyxl") as writer:
        used_df.to_excel(writer, sheet_name="lines", index=False)

    with pd.ExcelWriter(skipped_path, engine="openpyxl") as writer:
        skipped_df.to_excel(writer, sheet_name="lines", index=False)

    logger.info(
        "Exported discrete line reports: used=%s skipped=%s",
        used_path,
        skipped_path,
    )
    return [used_path, skipped_path]
