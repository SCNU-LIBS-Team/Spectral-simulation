from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

from modules.parsers import TargetSpec, parse_target_string


class ConfigValidationError(ValueError):
    """Raised when config.json is missing required fields or contains invalid values."""


@dataclass(frozen=True)
class DataDirsConfig:
    lines: str
    levels: str
    ionization_energies: str


@dataclass(frozen=True)
class SimulationConfig:
    temperature_K: float
    ne_cm3: float
    element_mole_fractions: Dict[str, float]
    wavelength_min_nm: float
    wavelength_max_nm: float
    delta_lambda_nm: float
    intensity_mode: Literal["energy", "photon"]
    broadening_mode: Literal["fixed", "stark"]
    fixed_fwhm_nm: float
    targets: List[TargetSpec]
    export_discrete_lines_used: bool
    data_dirs: DataDirsConfig
    output_dir: str


REQUIRED_CONFIG_FIELDS = (
    "temperature_K",
    "ne_cm3",
    "element_mole_fractions",
    "wavelength_min_nm",
    "wavelength_max_nm",
    "delta_lambda_nm",
    "intensity_mode",
    "broadening_mode",
    "fixed_fwhm_nm",
    "data_dirs",
    "output_dir",
)


def _require_fields(data: Dict[str, Any], field_names: tuple[str, ...]) -> None:
    for field_name in field_names:
        if field_name not in data:
            raise ConfigValidationError(
                f"Missing required config field '{field_name}' in config.json."
            )


def _validate_positive_number(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            f"Config field '{field_name}' must be a number."
        ) from exc

    if number <= 0.0:
        raise ConfigValidationError(
            f"Config field '{field_name}' must be positive."
        )
    return number


def _validate_element_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][a-z]?", symbol):
        raise ConfigValidationError(
            "element_mole_fractions keys must be standard chemical symbols such as Fe, Mn, Cr."
        )
    return symbol


def _validate_element_mole_fractions(data: Any) -> Dict[str, float]:
    if not isinstance(data, dict) or not data:
        raise ConfigValidationError(
            "Config field 'element_mole_fractions' must be a non-empty object."
        )

    normalized: Dict[str, float] = {}
    for key, value in data.items():
        symbol = _validate_element_symbol(key)
        fraction = _validate_positive_number(value, f"element_mole_fractions['{symbol}']")
        normalized[symbol] = fraction

    total = sum(normalized.values())
    if abs(total - 1.0) > 1.0e-6:
        raise ConfigValidationError(
            "The values in 'element_mole_fractions' must sum to 1 within a tolerance of 1e-6."
        )
    return normalized


def _validate_targets(data: Any) -> List[TargetSpec]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise ConfigValidationError("Config field 'targets' must be an array if provided.")

    parsed_targets: List[TargetSpec] = []
    seen_raw: set[str] = set()
    for item in data:
        if not isinstance(item, str):
            raise ConfigValidationError("Every target entry must be a string.")
        try:
            target = parse_target_string(item)
        except ValueError as exc:
            raise ConfigValidationError(
                f"Invalid target '{item}'. Expected formats like 'Fe' or 'Fe II'."
            ) from exc
        if target.raw in seen_raw:
            raise ConfigValidationError(f"Duplicate target '{target.raw}' was provided.")
        parsed_targets.append(target)
        seen_raw.add(target.raw)
    return parsed_targets


def _validate_data_dirs(data: Any) -> DataDirsConfig:
    if not isinstance(data, dict):
        raise ConfigValidationError("Config field 'data_dirs' must be an object.")

    required = ("lines", "levels", "ionization_energies")
    _require_fields(data, required)

    for key in required:
        if not isinstance(data[key], str) or not data[key].strip():
            raise ConfigValidationError(
                f"Config field 'data_dirs.{key}' must be a non-empty string path."
            )

    return DataDirsConfig(
        lines=data["lines"],
        levels=data["levels"],
        ionization_energies=data["ionization_energies"],
    )


def load_config(config_path: Path) -> SimulationConfig:
    if not config_path.exists():
        raise ConfigValidationError(f"Config file '{config_path}' does not exist.")

    try:
        raw_data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"config.json is not valid JSON: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigValidationError("config.json must contain a top-level JSON object.")

    _require_fields(raw_data, REQUIRED_CONFIG_FIELDS)

    temperature_K = _validate_positive_number(raw_data["temperature_K"], "temperature_K")
    ne_cm3 = _validate_positive_number(raw_data["ne_cm3"], "ne_cm3")
    wavelength_min_nm = _validate_positive_number(raw_data["wavelength_min_nm"], "wavelength_min_nm")
    wavelength_max_nm = _validate_positive_number(raw_data["wavelength_max_nm"], "wavelength_max_nm")
    delta_lambda_nm = _validate_positive_number(raw_data["delta_lambda_nm"], "delta_lambda_nm")
    fixed_fwhm_nm = _validate_positive_number(raw_data["fixed_fwhm_nm"], "fixed_fwhm_nm")

    if wavelength_max_nm <= wavelength_min_nm:
        raise ConfigValidationError(
            "Config field 'wavelength_max_nm' must be greater than 'wavelength_min_nm'."
        )

    intensity_mode = raw_data["intensity_mode"]
    if intensity_mode not in {"energy", "photon"}:
        raise ConfigValidationError(
            "Config field 'intensity_mode' must be either 'energy' or 'photon'."
        )

    broadening_mode = raw_data["broadening_mode"]
    if broadening_mode not in {"fixed", "stark"}:
        raise ConfigValidationError(
            "Config field 'broadening_mode' must be either 'fixed' or 'stark'."
        )

    output_dir = raw_data["output_dir"]
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigValidationError("Config field 'output_dir' must be a non-empty string.")

    export_discrete_lines_used = raw_data.get("export_discrete_lines_used", False)
    if not isinstance(export_discrete_lines_used, bool):
        raise ConfigValidationError(
            "Config field 'export_discrete_lines_used' must be true or false."
        )

    return SimulationConfig(
        temperature_K=temperature_K,
        ne_cm3=ne_cm3,
        element_mole_fractions=_validate_element_mole_fractions(raw_data["element_mole_fractions"]),
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        delta_lambda_nm=delta_lambda_nm,
        intensity_mode=intensity_mode,
        broadening_mode=broadening_mode,
        fixed_fwhm_nm=fixed_fwhm_nm,
        targets=_validate_targets(raw_data.get("targets", [])),
        export_discrete_lines_used=export_discrete_lines_used,
        data_dirs=_validate_data_dirs(raw_data["data_dirs"]),
        output_dir=output_dir,
    )
