from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from modules.config_loader import SimulationConfig
from modules.parsers import infer_element_from_filename, infer_stage_from_filename


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    lines_dir: Path
    levels_dir: Path
    ionization_energies_dir: Path
    output_dir: Path
    cleaned_data_dir: Path
    cleaned_lines_dir: Path
    cleaned_levels_dir: Path
    cleaned_ionization_energies_dir: Path
    log_file: Path


def resolve_project_paths(project_root: Path, config: SimulationConfig) -> ProjectPaths:
    lines_dir = (project_root / config.data_dirs.lines).resolve()
    levels_dir = (project_root / config.data_dirs.levels).resolve()
    ionization_energies_dir = (project_root / config.data_dirs.ionization_energies).resolve()
    output_dir = (project_root / config.output_dir).resolve()
    cleaned_data_dir = output_dir / "cleaned_data"
    cleaned_lines_dir = cleaned_data_dir / "lines"
    cleaned_levels_dir = cleaned_data_dir / "levels"
    cleaned_ionization_energies_dir = cleaned_data_dir / "ionization_energies"

    for required_dir in (lines_dir, levels_dir, ionization_energies_dir):
        if not required_dir.exists():
            raise FileNotFoundError(f"Required data directory does not exist: '{required_dir}'.")
        if not required_dir.is_dir():
            raise FileNotFoundError(f"Configured data path is not a directory: '{required_dir}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_lines_dir.mkdir(parents=True, exist_ok=True)
    cleaned_levels_dir.mkdir(parents=True, exist_ok=True)
    cleaned_ionization_energies_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        project_root=project_root.resolve(),
        lines_dir=lines_dir,
        levels_dir=levels_dir,
        ionization_energies_dir=ionization_energies_dir,
        output_dir=output_dir,
        cleaned_data_dir=cleaned_data_dir,
        cleaned_lines_dir=cleaned_lines_dir,
        cleaned_levels_dir=cleaned_levels_dir,
        cleaned_ionization_energies_dir=cleaned_ionization_energies_dir,
        log_file=output_dir / "run_log.txt",
    )


def _select_single_file(candidates: list[Path], description: str) -> Path:
    if not candidates:
        raise FileNotFoundError(f"Could not find {description}.")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(f"Multiple files matched {description}: {names}")
    return candidates[0]


def discover_line_file(lines_dir: Path, element: str) -> Path:
    candidates = []
    for path in lines_dir.glob("*.xlsx"):
        inferred_element = infer_element_from_filename(path.name)
        stem_lower = path.stem.lower()
        if inferred_element == element and "line" in stem_lower:
            candidates.append(path)
    return _select_single_file(candidates, f"line file for element '{element}' in '{lines_dir}'")


def discover_level_files(levels_dir: Path, element: str) -> Dict[int, Path]:
    discovered: Dict[int, Path] = {}
    for path in levels_dir.glob("*.xlsx"):
        inferred_element = infer_element_from_filename(path.name)
        inferred_stage = infer_stage_from_filename(path.name)
        stem_lower = path.stem.lower()
        if inferred_element != element or inferred_stage is None or "level" not in stem_lower:
            continue
        if inferred_stage in discovered:
            raise FileNotFoundError(
                f"Multiple levels files matched element '{element}' stage {inferred_stage}: "
                f"'{discovered[inferred_stage].name}' and '{path.name}'."
            )
        discovered[inferred_stage] = path
    return discovered


def discover_ionization_energy_file(ionization_dir: Path, element: str) -> Path:
    candidates = []
    for path in ionization_dir.glob("*.xlsx"):
        inferred_element = infer_element_from_filename(path.name)
        stem_lower = path.stem.lower()
        if inferred_element == element and ("energy" in stem_lower or "ionization" in stem_lower):
            candidates.append(path)
    return _select_single_file(
        candidates,
        f"ionization energy file for element '{element}' in '{ionization_dir}'",
    )
