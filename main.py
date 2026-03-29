from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from modules.config_loader import ConfigValidationError, SimulationConfig, load_config
from modules.data_reader import (
    DataFormatError,
    load_cleaned_table,
    read_ionization_energy_file,
    read_levels_file,
    read_lines_file,
    sanitize_ionization_energy_table,
    sanitize_levels_table,
    sanitize_lines_table,
)
from modules.exporter import export_cleaned_dataframe, export_continuous_spectra, export_discrete_line_tables
from modules.logger_utils import initialize_logger
from modules.path_manager import (
    ProjectPaths,
    discover_ionization_energy_file,
    discover_level_files,
    discover_line_file,
    resolve_project_paths,
)
from modules.physics import (
    UnitContext,
    build_ionization_energy_map,
    compute_partition_function,
    compute_saha_ion_fractions,
    validate_contiguous_stage_chain,
)
from modules.plotter import show_spectra
from modules.spectrum_builder import SpectrumBuildResult, build_spectra


def log_input_summary(logger, config: SimulationConfig, paths: ProjectPaths) -> None:
    logger.info("=== Input Summary ===")
    logger.info("temperature_K = %.6f", config.temperature_K)
    logger.info("ne_cm3 = %.6e (input electron density unit is cm^-3)", config.ne_cm3)
    logger.info("element_mole_fractions = %s", config.element_mole_fractions)
    logger.info("wavelength_min_nm = %.6f", config.wavelength_min_nm)
    logger.info("wavelength_max_nm = %.6f", config.wavelength_max_nm)
    logger.info("delta_lambda_nm = %.6f", config.delta_lambda_nm)
    logger.info("intensity_mode = %s", config.intensity_mode)
    logger.info("broadening_mode = %s", config.broadening_mode)
    logger.info("fixed_fwhm_nm = %.6f", config.fixed_fwhm_nm)
    logger.info(
        "data_dirs = {'lines': '%s', 'levels': '%s', 'ionization_energies': '%s'}",
        paths.lines_dir,
        paths.levels_dir,
        paths.ionization_energies_dir,
    )
    logger.info("output_dir = %s", paths.output_dir)
    logger.info("cleaned_data_dir = %s", paths.cleaned_data_dir)
    logger.info("targets = %s", [target.raw for target in config.targets])
    logger.info("export_discrete_lines_used = %s", config.export_discrete_lines_used)


def validate_targets_against_available_data(
    config: SimulationConfig,
    available_stages_by_element: Dict[str, List[int]],
) -> None:
    for target in config.targets:
        if target.element not in config.element_mole_fractions:
            raise ConfigValidationError(
                f"Target '{target.raw}' references element '{target.element}', "
                "but that element is not present in element_mole_fractions."
            )

        stages = available_stages_by_element.get(target.element, [])
        if target.kind == "ion" and target.ion_stage not in stages:
            raise ConfigValidationError(
                f"Target '{target.raw}' is invalid because ion stage {target.ion_stage} "
                f"for element '{target.element}' was not found in the available line data."
            )


def _extract_available_ionization_stages(
    element: str,
    ionization_energy_table: pd.DataFrame,
) -> List[int]:
    matching_rows = ionization_energy_table.loc[
        (ionization_energy_table["element"] == element)
        & ionization_energy_table["ion_stage"].notna()
        & ionization_energy_table["ionization_energy_eV"].notna()
    ]
    return sorted({int(stage) for stage in matching_rows["ion_stage"].tolist()})


def _select_supported_stage_chain(
    *,
    element: str,
    line_stages: List[int],
    level_stages: List[int],
    ionization_energy_stages: List[int],
    logger,
) -> List[int]:
    candidate_stages = sorted(set(line_stages) & set(level_stages))
    if not candidate_stages:
        raise FileNotFoundError(
            f"Could not find any ion stages for element '{element}' that are present in both "
            "the line data and the levels data."
        )

    ionization_stage_set = set(ionization_energy_stages)
    supported_stages = [candidate_stages[0]]

    while True:
        current_stage = supported_stages[-1]
        next_stage = current_stage + 1

        if next_stage not in candidate_stages:
            higher_candidates = [stage for stage in candidate_stages if stage > current_stage]
            if higher_candidates:
                logger.warning(
                    "Truncating the supported stage chain for %s after stage %d because stage %d is missing "
                    "from the combined line/levels data. Higher stages will be ignored: %s",
                    element,
                    current_stage,
                    next_stage,
                    higher_candidates,
                )
            break

        if current_stage not in ionization_stage_set:
            logger.warning(
                "Truncating the supported stage chain for %s after stage %d because the ionization energy "
                "for stage %d -> %d is missing.",
                element,
                current_stage,
                current_stage,
                next_stage,
            )
            break

        supported_stages.append(next_stage)

    return validate_contiguous_stage_chain(element, supported_stages)


def log_result_summary(
    logger,
    partition_functions: Dict[str, Dict[int, float]],
    ion_fractions: Dict[str, Dict[int, float]],
    build_result: SpectrumBuildResult,
    output_files: List[Path],
) -> None:
    logger.info("=== Result Summary ===")

    for element, stage_map in ion_fractions.items():
        for stage, fraction in stage_map.items():
            logger.info("Final ion fraction for %s stage %d = %.8e", element, stage, fraction)

    for element, stage_map in partition_functions.items():
        for stage, partition_function in stage_map.items():
            logger.info("Partition function for %s stage %d = %.8e", element, stage, partition_function)

    logger.info("Candidate line count = %d", build_result.candidate_line_count)
    logger.info("Used line count = %d", build_result.used_line_count)
    logger.info("Skipped line count = %d", build_result.skipped_line_count)
    logger.info(
        "Line count filtered by wavelength window = %d",
        build_result.filtered_by_wavelength_count,
    )
    logger.info("Output files = %s", [str(path) for path in output_files])


def prepare_element_models(
    config: SimulationConfig,
    unit_context: UnitContext,
    paths: ProjectPaths,
    logger,
) -> tuple[pd.DataFrame, Dict[str, Dict[int, float]], Dict[str, Dict[int, float]], Dict[str, List[int]]]:
    all_lines_frames: List[pd.DataFrame] = []
    partition_functions: Dict[str, Dict[int, float]] = {}
    ion_fractions: Dict[str, Dict[int, float]] = {}
    available_stages_by_element: Dict[str, List[int]] = {}

    for element in config.element_mole_fractions:
        line_file = discover_line_file(paths.lines_dir, element)
        ionization_energy_file = discover_ionization_energy_file(paths.ionization_energies_dir, element)
        level_files = discover_level_files(paths.levels_dir, element)

        raw_line_table = read_lines_file(line_file, logger)
        valid_line_stages = sorted({int(stage) for stage in raw_line_table["ion_stage"].dropna().tolist()})
        if not valid_line_stages:
            raise DataFormatError(
                f"No valid ion stages could be parsed from the line file '{line_file.name}' "
                f"for element '{element}'."
            )

        ionization_energy_table = read_ionization_energy_file(ionization_energy_file, logger)
        ionization_energy_stages = _extract_available_ionization_stages(element, ionization_energy_table)
        available_stages = _select_supported_stage_chain(
            element=element,
            line_stages=valid_line_stages,
            level_stages=sorted(level_files.keys()),
            ionization_energy_stages=ionization_energy_stages,
            logger=logger,
        )
        available_stages_by_element[element] = available_stages
        logger.info("Supported ion stage chain for %s = %s", element, available_stages)

        line_table = sanitize_lines_table(
            element=element,
            line_table=raw_line_table,
            supported_stages=available_stages,
            logger=logger,
        )
        cleaned_line_path = export_cleaned_dataframe(
            dataframe=line_table,
            output_path=paths.cleaned_lines_dir / line_file.name,
            sheet_name="lines",
            logger=logger,
        )
        line_table = load_cleaned_table(cleaned_line_path, logger)
        all_lines_frames.append(line_table)

        levels_by_stage: Dict[int, pd.DataFrame] = {}
        for stage in available_stages:
            if stage not in level_files:
                raise FileNotFoundError(
                    f"Missing levels file for element '{element}' ion stage {stage}. "
                    f"Expected a file like '{element}_one_Levels.xlsx' / '{element}_{stage}_Levels.xlsx'."
                )
            raw_levels_table = read_levels_file(level_files[stage], logger)
            levels_by_stage[stage] = sanitize_levels_table(
                element=element,
                ion_stage=stage,
                levels_table=raw_levels_table,
                logger=logger,
            )
            cleaned_levels_path = export_cleaned_dataframe(
                dataframe=levels_by_stage[stage],
                output_path=paths.cleaned_levels_dir / level_files[stage].name,
                sheet_name="levels",
                logger=logger,
            )
            levels_by_stage[stage] = load_cleaned_table(cleaned_levels_path, logger)

        ionization_energy_table = sanitize_ionization_energy_table(
            element=element,
            ionization_energy_table=ionization_energy_table,
            supported_stages=available_stages,
            logger=logger,
        )
        cleaned_ionization_path = export_cleaned_dataframe(
            dataframe=ionization_energy_table,
            output_path=paths.cleaned_ionization_energies_dir / ionization_energy_file.name,
            sheet_name="ionization",
            logger=logger,
        )
        ionization_energy_table = load_cleaned_table(cleaned_ionization_path, logger)

        ionization_energy_map = build_ionization_energy_map(
            element=element,
            available_stages=available_stages,
            ionization_energy_table=ionization_energy_table,
        )

        partition_functions[element] = {}
        for stage in available_stages:
            partition_functions[element][stage] = compute_partition_function(
                element=element,
                ion_stage=stage,
                levels_table=levels_by_stage[stage],
                unit_context=unit_context,
                logger=logger,
            )

        ion_fractions[element] = compute_saha_ion_fractions(
            element=element,
            available_stages=available_stages,
            partition_functions=partition_functions[element],
            ionization_energies_eV=ionization_energy_map,
            unit_context=unit_context,
            logger=logger,
        )

    all_lines = pd.concat(all_lines_frames, ignore_index=True) if all_lines_frames else pd.DataFrame()
    return all_lines, partition_functions, ion_fractions, available_stages_by_element


def run() -> int:
    project_root = Path(__file__).resolve().parent
    bootstrap_log_file = project_root / "output" / "run_log.txt"
    logger = initialize_logger(bootstrap_log_file)

    try:
        config = load_config(project_root / "config.json")
        paths = resolve_project_paths(project_root, config)
        if paths.log_file != bootstrap_log_file:
            logger = initialize_logger(paths.log_file)
        log_input_summary(logger, config, paths)

        unit_context = UnitContext.from_inputs(
            temperature_K=config.temperature_K,
            ne_cm3=config.ne_cm3,
            logger=logger,
        )

        all_lines, partition_functions, ion_fractions, available_stages_by_element = prepare_element_models(
            config=config,
            unit_context=unit_context,
            paths=paths,
            logger=logger,
        )

        validate_targets_against_available_data(config, available_stages_by_element)

        build_result = build_spectra(
            config=config,
            lines_table=all_lines,
            partition_functions=partition_functions,
            ion_fractions=ion_fractions,
            unit_context=unit_context,
            logger=logger,
        )

        output_files = export_continuous_spectra(
            build_result=build_result,
            config=config,
            output_dir=paths.output_dir,
            logger=logger,
        )

        if config.export_discrete_lines_used:
            output_files.extend(
                export_discrete_line_tables(
                    build_result=build_result,
                    output_dir=paths.output_dir,
                    logger=logger,
                )
            )

        log_result_summary(
            logger=logger,
            partition_functions=partition_functions,
            ion_fractions=ion_fractions,
            build_result=build_result,
            output_files=output_files,
        )

        show_spectra(build_result=build_result, config=config, logger=logger)
        return 0

    except (ConfigValidationError, DataFormatError, FileNotFoundError, NotImplementedError, ValueError) as exc:
        logger.error("Simulation failed: %s", exc)
        print(f"Simulation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        logger.exception("Unexpected failure: %s", exc)
        print(f"Unexpected failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
