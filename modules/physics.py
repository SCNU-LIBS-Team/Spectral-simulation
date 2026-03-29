from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


PLANCK_CONSTANT = 6.62607015e-34
LIGHT_SPEED = 2.99792458e8
BOLTZMANN_CONSTANT = 1.380649e-23
ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31


@dataclass(frozen=True)
class UnitContext:
    temperature_K: float
    ne_cm3: float
    ne_m3: float
    kbt_J: float
    kbt_eV: float

    @classmethod
    def from_inputs(cls, temperature_K: float, ne_cm3: float, logger) -> "UnitContext":
        ne_m3 = ne_cm3 * 1.0e6
        kbt_J = BOLTZMANN_CONSTANT * temperature_K
        kbt_eV = kbt_J / ELEMENTARY_CHARGE

        logger.info(
            "Unit conversion: ne_cm3 = %.6e cm^-3 converted to ne_m3 = %.6e m^-3 for Saha calculations.",
            ne_cm3,
            ne_m3,
        )
        logger.info(
            "Unit conversion: k_B T = %.6e J = %.6e eV at T = %.6f K.",
            kbt_J,
            kbt_eV,
            temperature_K,
        )

        return cls(
            temperature_K=temperature_K,
            ne_cm3=ne_cm3,
            ne_m3=ne_m3,
            kbt_J=kbt_J,
            kbt_eV=kbt_eV,
        )

    @staticmethod
    def wavenumber_cm1_to_j(value_cm1: float | np.ndarray) -> float | np.ndarray:
        return PLANCK_CONSTANT * LIGHT_SPEED * 100.0 * value_cm1

    @staticmethod
    def wavenumber_cm1_to_eV(value_cm1: float | np.ndarray) -> float | np.ndarray:
        return UnitContext.wavenumber_cm1_to_j(value_cm1) / ELEMENTARY_CHARGE

    @staticmethod
    def ev_to_j(value_eV: float | np.ndarray) -> float | np.ndarray:
        return value_eV * ELEMENTARY_CHARGE

    @staticmethod
    def nm_to_m(value_nm: float | np.ndarray) -> float | np.ndarray:
        return value_nm * 1.0e-9


def validate_contiguous_stage_chain(element: str, stages: Iterable[int]) -> List[int]:
    sorted_stages = sorted(set(int(stage) for stage in stages))
    if not sorted_stages:
        raise ValueError(f"No ion stages are available for element '{element}'.")

    for current_stage, next_stage in zip(sorted_stages, sorted_stages[1:]):
        if next_stage != current_stage + 1:
            raise ValueError(
                f"Element '{element}' has a non-contiguous ion stage chain: {sorted_stages}. "
                f"Missing stage {current_stage + 1} between {current_stage} and {next_stage}."
            )
    return sorted_stages


def compute_partition_function(
    element: str,
    ion_stage: int,
    levels_table: pd.DataFrame,
    unit_context: UnitContext,
    logger,
) -> float:
    valid_rows = []
    seen_pairs = set()

    for _, row in levels_table.iterrows():
        j_value = row.get("J")
        level_cm1 = row.get("level_cm1")
        if j_value is None or level_cm1 is None or pd.isna(j_value) or pd.isna(level_cm1):
            logger.warning(
                "Skipping level row in %s stage %d at source row %s because J or level_cm1 is missing/invalid.",
                element,
                ion_stage,
                row.get("source_row_number"),
            )
            continue

        key = (float(j_value), float(level_cm1))
        if key in seen_pairs:
            logger.warning(
                "Skipping duplicate level row in %s stage %d for (J, level_cm1) = (%s, %s).",
                element,
                ion_stage,
                j_value,
                level_cm1,
            )
            continue

        seen_pairs.add(key)
        valid_rows.append(key)

    if not valid_rows:
        raise ValueError(
            f"No valid levels remain for element '{element}' ion stage {ion_stage}; "
            "cannot build the partition function."
        )

    j_values = np.asarray([item[0] for item in valid_rows], dtype=float)
    level_cm1_values = np.asarray([item[1] for item in valid_rows], dtype=float)
    degeneracies = 2.0 * j_values + 1.0
    energies_J = UnitContext.wavenumber_cm1_to_j(level_cm1_values)

    partition_function = float(np.sum(degeneracies * np.exp(-energies_J / unit_context.kbt_J)))
    if partition_function <= 0.0:
        raise ValueError(
            f"Partition function became non-positive for element '{element}' ion stage {ion_stage}."
        )

    logger.info(
        "Partition function U(T) for %s stage %d at %.3f K = %.8e",
        element,
        ion_stage,
        unit_context.temperature_K,
        partition_function,
    )
    return partition_function


def build_ionization_energy_map(
    element: str,
    available_stages: List[int],
    ionization_energy_table: pd.DataFrame,
) -> Dict[int, float]:
    energy_map: Dict[int, float] = {}

    for _, row in ionization_energy_table.iterrows():
        if row.get("element") != element:
            continue
        ion_stage = row.get("ion_stage")
        energy_eV = row.get("ionization_energy_eV")
        if ion_stage is None or energy_eV is None or pd.isna(ion_stage) or pd.isna(energy_eV):
            continue
        energy_map[int(ion_stage)] = float(energy_eV)

    for stage in available_stages[:-1]:
        if stage not in energy_map:
            raise ValueError(
                f"Ionization energy chain is incomplete for element '{element}': "
                f"missing ionization energy for stage {stage} -> {stage + 1}."
            )
    return energy_map


def compute_saha_ion_fractions(
    element: str,
    available_stages: List[int],
    partition_functions: Dict[int, float],
    ionization_energies_eV: Dict[int, float],
    unit_context: UnitContext,
    logger,
) -> Dict[int, float]:
    ratios = [1.0]
    saha_prefactor = ((2.0 * math.pi * ELECTRON_MASS * BOLTZMANN_CONSTANT * unit_context.temperature_K) / (PLANCK_CONSTANT**2)) ** 1.5
    saha_prefactor /= unit_context.ne_m3

    for current_stage, next_stage in zip(available_stages, available_stages[1:]):
        current_partition = partition_functions[current_stage]
        next_partition = partition_functions[next_stage]
        ionization_energy_J = UnitContext.ev_to_j(ionization_energies_eV[current_stage])
        ratio = saha_prefactor * 2.0 * (next_partition / current_partition) * math.exp(
            -ionization_energy_J / unit_context.kbt_J
        )
        ratios.append(ratios[-1] * ratio)

    normalization = sum(ratios)
    fractions = {stage: weight / normalization for stage, weight in zip(available_stages, ratios)}

    for stage in available_stages:
        logger.info("Saha fraction for %s stage %d = %.8e", element, stage, fractions[stage])

    logger.warning(
        "Saha truncation note for %s: the highest included ion stage is truncated in v1.0, "
        "so that stage can be overestimated at high temperature.",
        element,
    )
    if available_stages[0] > 1:
        logger.warning(
            "Saha truncation note for %s: stages below %d are not present in the current data set, "
            "so fractions are renormalized only across the included stages.",
            element,
            available_stages[0],
        )
    return fractions


def compute_line_intensity(
    wavelength_nm: float,
    Aki: float,
    Ek_cm1: float,
    J_upper: float,
    element_mole_fraction: float,
    ion_fraction: float,
    partition_function: float,
    intensity_mode: str,
    unit_context: UnitContext,
) -> tuple[float, float]:
    g_upper = 2.0 * J_upper + 1.0
    boltzmann_factor = math.exp(-float(UnitContext.wavenumber_cm1_to_j(Ek_cm1)) / unit_context.kbt_J)
    line_intensity = Aki * g_upper * element_mole_fraction * ion_fraction * boltzmann_factor / partition_function

    if intensity_mode == "energy":
        photon_energy = PLANCK_CONSTANT * LIGHT_SPEED / float(UnitContext.nm_to_m(wavelength_nm))
        line_intensity *= photon_energy
    elif intensity_mode != "photon":
        raise ValueError(f"Unsupported intensity_mode '{intensity_mode}'.")

    return line_intensity, g_upper
