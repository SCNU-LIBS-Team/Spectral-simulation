from __future__ import annotations

import math
from typing import Any

import numpy as np


def apply_line_broadening(
    *,
    broadening_mode: str,
    wavelength_grid_nm: np.ndarray,
    center_wavelength_nm: float,
    line_area: float,
    fixed_fwhm_nm: float,
    line_record: dict[str, Any],
    config: Any,
) -> np.ndarray:
    if broadening_mode == "fixed":
        return apply_fixed_lorentzian(
            wavelength_grid_nm=wavelength_grid_nm,
            center_wavelength_nm=center_wavelength_nm,
            line_area=line_area,
            fwhm_nm=fixed_fwhm_nm,
        )
    if broadening_mode == "stark":
        return apply_stark_broadening(
            wavelength_grid_nm=wavelength_grid_nm,
            center_wavelength_nm=center_wavelength_nm,
            line_area=line_area,
            line_record=line_record,
            config=config,
        )
    raise ValueError(f"Unsupported broadening_mode '{broadening_mode}'.")


def apply_fixed_lorentzian(
    *,
    wavelength_grid_nm: np.ndarray,
    center_wavelength_nm: float,
    line_area: float,
    fwhm_nm: float,
) -> np.ndarray:
    gamma_nm = fwhm_nm / 2.0
    left_nm = center_wavelength_nm - 20.0 * fwhm_nm
    right_nm = center_wavelength_nm + 20.0 * fwhm_nm

    profile = np.zeros_like(wavelength_grid_nm, dtype=float)
    mask = (wavelength_grid_nm >= left_nm) & (wavelength_grid_nm <= right_nm)
    if not np.any(mask):
        return profile

    offsets = wavelength_grid_nm[mask] - center_wavelength_nm
    lorentzian = (1.0 / math.pi) * gamma_nm / (offsets**2 + gamma_nm**2)
    analytic_integral = (
        math.atan((right_nm - center_wavelength_nm) / gamma_nm)
        - math.atan((left_nm - center_wavelength_nm) / gamma_nm)
    ) / math.pi

    if analytic_integral <= 0.0:
        raise ValueError("Truncated Lorentzian analytic normalization became non-positive.")

    profile[mask] = line_area * (lorentzian / analytic_integral)
    return profile


def apply_stark_broadening(
    *,
    wavelength_grid_nm: np.ndarray,
    center_wavelength_nm: float,
    line_area: float,
    line_record: dict[str, Any],
    config: Any,
) -> np.ndarray:
    raise NotImplementedError(
        "Stark broadening is reserved for a future version and is not implemented in v1.0."
    )


def apply_instrument_broadening(*args: Any, **kwargs: Any) -> np.ndarray:
    raise NotImplementedError(
        "Instrument broadening is reserved for a future version and is not implemented in v1.0."
    )


def apply_voigt_broadening(*args: Any, **kwargs: Any) -> np.ndarray:
    raise NotImplementedError(
        "Voigt broadening is reserved for a future version and is not implemented in v1.0."
    )


def apply_self_absorption_correction(*args: Any, **kwargs: Any) -> np.ndarray:
    raise NotImplementedError(
        "Self-absorption correction is reserved for a future version and is not implemented in v1.0."
    )
