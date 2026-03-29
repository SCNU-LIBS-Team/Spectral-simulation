from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from modules.config_loader import SimulationConfig
from modules.spectrum_builder import SpectrumBuildResult


def _normalized_values(raw_values: np.ndarray, total_max_raw: float) -> np.ndarray:
    if total_max_raw <= 0.0:
        return np.zeros_like(raw_values)
    return raw_values / total_max_raw


def _relative_max_difference(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    if reference_scale <= 0.0:
        return 0.0
    return float(np.max(np.abs(reference - candidate)) / reference_scale)


def _integrated_intensity(raw_values: np.ndarray, wavelength_grid_nm: np.ndarray) -> float:
    if raw_values.size < 2:
        return float(np.sum(raw_values))
    return float(np.trapz(raw_values, wavelength_grid_nm))


def show_spectra(*, build_result: SpectrumBuildResult, config: SimulationConfig, logger) -> None:
    total_raw = build_result.spectra_raw["total"]
    total_max_raw = float(np.max(total_raw)) if total_raw.size else 0.0
    total_integral_raw = _integrated_intensity(total_raw, build_result.wavelength_grid_nm)

    logger.info("Showing total spectrum")
    plt.figure()
    plt.plot(
        build_result.wavelength_grid_nm,
        _normalized_values(total_raw, total_max_raw),
        linewidth=1.0,
    )
    plt.title("Total Spectrum (norm)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("norm")
    plt.tight_layout()

    single_element = None
    if len(config.element_mole_fractions) == 1:
        single_element = next(iter(config.element_mole_fractions))

    for target in config.targets:
        if (
            single_element is not None
            and target.kind == "element"
            and target.element == single_element
        ):
            logger.info(
                "Skipping target spectrum '%s' because it is mathematically identical to the total spectrum "
                "when only one element (%s) is present in element_mole_fractions.",
                target.raw,
                single_element,
            )
            continue

        target_raw = build_result.spectra_raw[target.raw]
        target_peak_ratio = float(np.max(target_raw) / total_max_raw) if total_max_raw > 0.0 else 0.0
        target_integral_raw = _integrated_intensity(target_raw, build_result.wavelength_grid_nm)
        target_integral_ratio = target_integral_raw / total_integral_raw if total_integral_raw > 0.0 else 0.0
        relative_difference = _relative_max_difference(total_raw, target_raw)

        logger.info(
            "Target spectrum diagnostics for %s: peak_ratio_to_total = %.6f, integral_ratio_to_total = %.6f, "
            "relative_max_difference_from_total = %.6e",
            target.raw,
            target_peak_ratio,
            target_integral_ratio,
            relative_difference,
        )
        if target_peak_ratio > 0.99 and target_integral_ratio > 0.90:
            logger.warning(
                "Target spectrum '%s' is expected to look very similar to the total spectrum because it carries "
                "most of the peak and integrated intensity under the current plasma conditions.",
                target.raw,
            )

        logger.info("Showing target spectrum: %s", target.raw)
        plt.figure()
        plt.plot(
            build_result.wavelength_grid_nm,
            _normalized_values(target_raw, total_max_raw),
            linewidth=1.0,
        )
        plt.title(
            f"{target.raw} Spectrum (norm)\n"
            f"peak/total={target_peak_ratio:.3f}, integral/total={target_integral_ratio:.3f}"
        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("norm")
        plt.tight_layout()

    plt.show()
