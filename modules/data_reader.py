from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from modules.parsers import (
    infer_element_from_filename,
    infer_stage_from_filename,
    normalize_header_text,
    parse_element_symbol,
    parse_ion_label,
    parse_ionization_stage,
    parse_j_value,
    parse_numeric_cell,
    standardize_header_names,
)


class DataFormatError(ValueError):
    """Raised when an input workbook cannot be parsed into the required internal schema."""


@dataclass(frozen=True)
class TableLayout:
    sheet_name: str
    header_start_row: int
    header_height: int
    matched_columns: Dict[int, str]
    original_headers: Dict[str, str]
    score: float


@dataclass(frozen=True)
class StandardizedTable:
    dataframe: pd.DataFrame
    sheet_name: str
    original_headers: Dict[str, str]


MINIMUM_LAYOUT_SCORE = {
    "lines": 6.0,
    "levels": 2.0,
    "ionization": 2.0,
}

HEADER_HEIGHT_OPTIONS = (1, 2, 3, 4)
PREVIEW_ROW_LIMIT = 80
DATA_QUALITY_SAMPLE_ROWS = 16


def _is_blank_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return not text or text.lower() == "nan"


def _clean_cell_value(value: object) -> object:
    if _is_blank_value(value):
        return None
    return value


def _combine_header_rows(header_rows: pd.DataFrame) -> list[str]:
    combined_headers: list[str] = []
    for column_index in range(header_rows.shape[1]):
        pieces = []
        for value in header_rows.iloc[:, column_index].tolist():
            cleaned = _clean_cell_value(value)
            if cleaned is None:
                continue
            pieces.append(str(cleaned).strip())
        combined_headers.append(" ".join(pieces).strip())
    return combined_headers


def _forward_fill_header_rows(header_rows: pd.DataFrame) -> pd.DataFrame:
    filled = header_rows.copy()
    for row_index in range(filled.shape[0]):
        last_value: object | None = None
        for column_index in range(filled.shape[1]):
            value = filled.iat[row_index, column_index]
            cleaned = _clean_cell_value(value)
            if cleaned is None:
                if last_value is not None:
                    filled.iat[row_index, column_index] = last_value
            else:
                last_value = cleaned
                filled.iat[row_index, column_index] = cleaned
    return filled


def _header_contains_tokens(header_text: str, *tokens: str) -> bool:
    normalized = normalize_header_text(header_text)
    return all(token in normalized for token in tokens)


def _parse_success_count(series: pd.Series, parser) -> int:
    count = 0
    for value in series.tolist():
        cleaned = _clean_cell_value(value)
        if cleaned is None:
            continue
        try:
            parsed = parser(cleaned)
        except ValueError:
            continue
        if parsed is None or (isinstance(parsed, float) and pd.isna(parsed)):
            continue
        count += 1
    return count


def _pick_best_parseable_column(candidate_indexes: list[int], data_preview: pd.DataFrame, parser) -> Optional[int]:
    best_index: Optional[int] = None
    best_score = 0

    for column_index in candidate_indexes:
        if column_index >= data_preview.shape[1]:
            continue
        score = _parse_success_count(data_preview.iloc[:, column_index], parser)
        if score > best_score:
            best_index = column_index
            best_score = score

    return best_index


def _match_lines_columns(
    header_rows: pd.DataFrame,
    data_preview: pd.DataFrame,
) -> tuple[Dict[int, str], Dict[str, str]]:
    raw_headers = _combine_header_rows(header_rows)
    matched_columns = standardize_header_names(raw_headers, "lines")
    original_headers = {canonical: raw_headers[index] for index, canonical in matched_columns.items()}

    filled_headers = _combine_header_rows(_forward_fill_header_rows(header_rows))
    lower_group_columns = [
        index for index, header in enumerate(filled_headers) if _header_contains_tokens(header, "lower", "level")
    ]
    upper_group_columns = [
        index for index, header in enumerate(filled_headers) if _header_contains_tokens(header, "upper", "level")
    ]

    if "J_lower" not in matched_columns.values():
        lower_index = _pick_best_parseable_column(lower_group_columns, data_preview, parse_j_value)
        if lower_index is not None:
            matched_columns[lower_index] = "J_lower"
            original_headers["J_lower"] = filled_headers[lower_index]

    if "J_upper" not in matched_columns.values():
        upper_index = _pick_best_parseable_column(upper_group_columns, data_preview, parse_j_value)
        if upper_index is not None:
            matched_columns[upper_index] = "J_upper"
            original_headers["J_upper"] = filled_headers[upper_index]

    return matched_columns, original_headers


def _match_columns_for_layout(
    header_rows: pd.DataFrame,
    data_preview: pd.DataFrame,
    dataset_kind: str,
) -> tuple[Dict[int, str], Dict[str, str]]:
    if dataset_kind == "lines":
        return _match_lines_columns(header_rows, data_preview)

    headers = _combine_header_rows(header_rows)
    matched_columns = standardize_header_names(headers, dataset_kind)
    original_headers = {canonical: headers[index] for index, canonical in matched_columns.items()}
    return matched_columns, original_headers


def _score_layout(dataset_kind: str, matched_columns: Dict[int, str]) -> float:
    matched = set(matched_columns.values())
    score = float(len(matched))

    if dataset_kind == "lines":
        if "ion_label" in matched:
            score += 2.0
        if "Aki" in matched:
            score += 2.0
        if {"wavelength_observed_air_nm", "wavelength_ritz_air_nm"} & matched:
            score += 2.0
        if "Ek_cm1" in matched:
            score += 1.0
        if "J_upper" in matched:
            score += 1.0
    elif dataset_kind == "levels":
        if {"J", "level_cm1"} <= matched:
            score += 1.0
    elif dataset_kind == "ionization":
        stage_fields = {"ion_stage_token", "ion_charge_token", "species_name"}
        if matched & stage_fields and "ionization_energy_eV" in matched:
            score += 1.0

    return score


def _count_valid_line_rows(data_preview: pd.DataFrame, reverse_columns: Dict[str, int]) -> int:
    required_columns = {"ion_label", "Aki", "Ek_cm1"}
    if not required_columns <= set(reverse_columns):
        return 0

    valid_rows = 0
    for _, row in data_preview.iterrows():
        values = row.tolist()
        if all(_is_blank_value(value) for value in values):
            continue

        ion_raw = row.iloc[reverse_columns["ion_label"]]
        try:
            parse_ion_label(ion_raw)
        except ValueError:
            continue

        wavelength_observed = None
        wavelength_ritz = None
        if "wavelength_observed_air_nm" in reverse_columns:
            wavelength_observed = parse_numeric_cell(row.iloc[reverse_columns["wavelength_observed_air_nm"]])
        if "wavelength_ritz_air_nm" in reverse_columns:
            wavelength_ritz = parse_numeric_cell(row.iloc[reverse_columns["wavelength_ritz_air_nm"]])

        if wavelength_observed is None and wavelength_ritz is None:
            continue
        if parse_numeric_cell(row.iloc[reverse_columns["Aki"]]) is None:
            continue
        if parse_numeric_cell(row.iloc[reverse_columns["Ek_cm1"]]) is None:
            continue
        if "J_upper" in reverse_columns:
            try:
                parse_j_value(row.iloc[reverse_columns["J_upper"]])
            except ValueError:
                continue

        valid_rows += 1

    return valid_rows


def _count_valid_level_rows(data_preview: pd.DataFrame, reverse_columns: Dict[str, int]) -> int:
    if not {"J", "level_cm1"} <= set(reverse_columns):
        return 0

    valid_rows = 0
    for _, row in data_preview.iterrows():
        values = row.tolist()
        if all(_is_blank_value(value) for value in values):
            continue

        try:
            j_value = parse_j_value(row.iloc[reverse_columns["J"]])
        except ValueError:
            continue
        level_value = parse_numeric_cell(row.iloc[reverse_columns["level_cm1"]])
        if pd.isna(j_value) or level_value is None:
            continue
        valid_rows += 1

    return valid_rows


def _count_valid_ionization_rows(
    data_preview: pd.DataFrame,
    reverse_columns: Dict[str, int],
    original_headers: Dict[str, str],
) -> int:
    if "ionization_energy_eV" not in reverse_columns:
        return 0

    stage_priority = ("ion_charge_token", "ion_stage_token", "species_name")
    valid_rows = 0
    for _, row in data_preview.iterrows():
        values = row.tolist()
        if all(_is_blank_value(value) for value in values):
            continue

        stage_value = None
        stage_hint = ""
        for canonical_name in stage_priority:
            column_index = reverse_columns.get(canonical_name)
            if column_index is None:
                continue
            candidate = row.iloc[column_index]
            if _clean_cell_value(candidate) is None:
                continue
            stage_value = candidate
            stage_hint = original_headers.get(canonical_name, "")
            break

        if stage_value is None:
            continue
        try:
            stage = parse_ionization_stage(stage_value, stage_hint)
        except ValueError:
            continue
        if stage is None:
            continue

        if parse_numeric_cell(row.iloc[reverse_columns["ionization_energy_eV"]]) is None:
            continue
        valid_rows += 1

    return valid_rows


def _score_candidate_data(
    dataset_kind: str,
    data_preview: pd.DataFrame,
    matched_columns: Dict[int, str],
    original_headers: Dict[str, str],
) -> float:
    reverse_columns = {canonical: index for index, canonical in matched_columns.items()}

    if dataset_kind == "lines":
        valid_rows = _count_valid_line_rows(data_preview, reverse_columns)
        j_bonus = 0.5 if "J_lower" in reverse_columns else 0.0
        j_bonus += 0.5 if "J_upper" in reverse_columns else 0.0
        return float(valid_rows) * 2.0 + j_bonus

    if dataset_kind == "levels":
        return float(_count_valid_level_rows(data_preview, reverse_columns)) * 2.0

    if dataset_kind == "ionization":
        return float(_count_valid_ionization_rows(data_preview, reverse_columns, original_headers)) * 2.0

    return 0.0


def _detect_table_layout(path: Path, dataset_kind: str) -> TableLayout:
    excel_file = pd.ExcelFile(path, engine="openpyxl")
    best_layout: Optional[TableLayout] = None

    for sheet_name in excel_file.sheet_names:
        preview = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=None,
            engine="openpyxl",
            dtype=object,
            nrows=PREVIEW_ROW_LIMIT,
        )
        if preview.empty:
            continue

        preview = preview.dropna(axis=1, how="all")
        if preview.empty:
            continue

        max_start = min(50, len(preview))
        for header_start_row in range(max_start):
            for header_height in HEADER_HEIGHT_OPTIONS:
                data_start_row = header_start_row + header_height
                if data_start_row >= len(preview):
                    continue

                header_rows = preview.iloc[header_start_row:data_start_row, :]
                data_preview = preview.iloc[data_start_row : data_start_row + DATA_QUALITY_SAMPLE_ROWS, :]

                matched_columns, original_headers = _match_columns_for_layout(
                    header_rows=header_rows,
                    data_preview=data_preview,
                    dataset_kind=dataset_kind,
                )
                if not matched_columns:
                    continue

                layout_score = _score_layout(dataset_kind, matched_columns)
                if layout_score < MINIMUM_LAYOUT_SCORE[dataset_kind]:
                    continue

                total_score = layout_score + _score_candidate_data(
                    dataset_kind=dataset_kind,
                    data_preview=data_preview,
                    matched_columns=matched_columns,
                    original_headers=original_headers,
                )

                candidate = TableLayout(
                    sheet_name=sheet_name,
                    header_start_row=header_start_row,
                    header_height=header_height,
                    matched_columns=matched_columns,
                    original_headers=original_headers,
                    score=total_score,
                )
                if best_layout is None or candidate.score > best_layout.score:
                    best_layout = candidate

    if best_layout is None:
        raise DataFormatError(
            f"Could not automatically detect a valid {dataset_kind} table in workbook '{path.name}'."
        )
    return best_layout


def _read_standardized_table(path: Path, dataset_kind: str) -> StandardizedTable:
    layout = _detect_table_layout(path, dataset_kind)
    full_sheet = pd.read_excel(
        path,
        sheet_name=layout.sheet_name,
        header=None,
        engine="openpyxl",
        dtype=object,
    )
    full_sheet = full_sheet.dropna(axis=1, how="all")

    data = full_sheet.iloc[layout.header_start_row + layout.header_height :, :].copy()
    if data.empty:
        raise DataFormatError(
            f"Detected header rows in '{path.name}' sheet '{layout.sheet_name}', but no data rows were found."
        )

    original_excel_rows = pd.Series(data.index + 1).reset_index(drop=True)
    data = data.reset_index(drop=True)

    standardized_columns: Dict[str, pd.Series] = {}
    for column_index, canonical_name in layout.matched_columns.items():
        if canonical_name in standardized_columns or column_index >= data.shape[1]:
            continue
        standardized_columns[canonical_name] = data.iloc[:, column_index].map(_clean_cell_value)

    standardized = pd.DataFrame(standardized_columns)
    non_empty_mask = standardized.apply(lambda row: any(value is not None for value in row.tolist()), axis=1)
    standardized = standardized.loc[non_empty_mask].reset_index(drop=True)
    original_excel_rows = original_excel_rows.loc[non_empty_mask].reset_index(drop=True)

    standardized["source_file"] = path.name
    standardized["source_sheet"] = layout.sheet_name
    standardized["source_row_number"] = original_excel_rows
    return StandardizedTable(
        dataframe=standardized,
        sheet_name=layout.sheet_name,
        original_headers=layout.original_headers,
    )


def _safe_parse_j(value: object) -> tuple[Optional[float], Optional[str]]:
    try:
        parsed = parse_j_value(value)
    except ValueError as exc:
        return None, str(exc)

    if pd.isna(parsed):
        return None, "Missing J value."
    return float(parsed), None


def read_lines_file(path: Path, logger) -> pd.DataFrame:
    table = _read_standardized_table(path, "lines")
    dataframe = table.dataframe
    required_columns = {"ion_label", "Aki", "Ei_cm1", "Ek_cm1", "J_lower", "J_upper"}
    missing = sorted(required_columns - set(dataframe.columns))
    if missing:
        raise DataFormatError(
            f"Line file '{path.name}' is missing required columns after header normalization: {missing}"
        )
    if "wavelength_observed_air_nm" not in dataframe.columns and "wavelength_ritz_air_nm" not in dataframe.columns:
        raise DataFormatError(
            f"Line file '{path.name}' does not contain an observed or Ritz wavelength column."
        )

    source_element = infer_element_from_filename(path.name)
    if source_element is None:
        raise DataFormatError(
            f"Could not infer the element symbol from line file name '{path.name}'."
        )

    records = []
    for _, row in dataframe.iterrows():
        ion_raw = row.get("ion_label")
        element = source_element
        ion_stage = None
        ion_label = str(ion_raw).strip() if ion_raw is not None else None
        ion_parse_error = None
        if ion_raw is None or str(ion_raw).strip() == "":
            ion_parse_error = "Missing Ion label."
        else:
            try:
                ion_identity = parse_ion_label(ion_raw, expected_element=source_element)
                element = ion_identity.element
                ion_stage = ion_identity.ion_stage
                ion_label = ion_identity.ion_label
            except ValueError as exc:
                ion_parse_error = str(exc)

        j_lower_value, j_lower_error = _safe_parse_j(row.get("J_lower"))
        j_upper_value, j_upper_error = _safe_parse_j(row.get("J_upper"))

        records.append(
            {
                "element": element,
                "ion_label": ion_label,
                "ion_stage": ion_stage,
                "wavelength_observed_air_nm": parse_numeric_cell(row.get("wavelength_observed_air_nm")),
                "wavelength_ritz_air_nm": parse_numeric_cell(row.get("wavelength_ritz_air_nm")),
                "Aki": parse_numeric_cell(row.get("Aki")),
                "Ei_cm1": parse_numeric_cell(row.get("Ei_cm1")),
                "Ek_cm1": parse_numeric_cell(row.get("Ek_cm1")),
                "J_lower": j_lower_value,
                "J_upper": j_upper_value,
                "J_lower_raw": row.get("J_lower"),
                "J_upper_raw": row.get("J_upper"),
                "ion_parse_error": ion_parse_error,
                "J_lower_parse_error": j_lower_error,
                "J_upper_parse_error": j_upper_error,
                "source_file": row.get("source_file"),
                "source_sheet": row.get("source_sheet"),
                "source_row_number": row.get("source_row_number"),
            }
        )

    logger.info(
        "Parsed line file '%s' from sheet '%s' with %d rows.",
        path.name,
        table.sheet_name,
        len(records),
    )
    return pd.DataFrame(records)


def read_levels_file(path: Path, logger) -> pd.DataFrame:
    table = _read_standardized_table(path, "levels")
    dataframe = table.dataframe
    required_columns = {"J", "level_cm1"}
    missing = sorted(required_columns - set(dataframe.columns))
    if missing:
        raise DataFormatError(
            f"Levels file '{path.name}' is missing required columns after header normalization: {missing}"
        )

    source_element = infer_element_from_filename(path.name)
    source_stage = infer_stage_from_filename(path.name)
    if source_element is None or source_stage is None:
        raise DataFormatError(
            f"Could not infer element or ion stage from levels file name '{path.name}'."
        )

    records = []
    for _, row in dataframe.iterrows():
        j_value, j_error = _safe_parse_j(row.get("J"))
        records.append(
            {
                "element": source_element,
                "ion_stage": source_stage,
                "J": j_value,
                "level_cm1": parse_numeric_cell(row.get("level_cm1")),
                "J_raw": row.get("J"),
                "J_parse_error": j_error,
                "source_file": row.get("source_file"),
                "source_sheet": row.get("source_sheet"),
                "source_row_number": row.get("source_row_number"),
            }
        )

    logger.info(
        "Parsed levels file '%s' from sheet '%s' with %d rows.",
        path.name,
        table.sheet_name,
        len(records),
    )
    return pd.DataFrame(records)


def read_ionization_energy_file(path: Path, logger) -> pd.DataFrame:
    table = _read_standardized_table(path, "ionization")
    dataframe = table.dataframe
    stage_columns = {"ion_stage_token", "ion_charge_token", "species_name"} & set(dataframe.columns)
    if not stage_columns:
        raise DataFormatError(
            f"Ionization energy file '{path.name}' is missing a readable ion-stage column after header normalization."
        )
    if "ionization_energy_eV" not in dataframe.columns:
        raise DataFormatError(
            f"Ionization energy file '{path.name}' is missing required columns after header normalization: "
            "['ionization_energy_eV']"
        )

    source_element = infer_element_from_filename(path.name)
    if source_element is None:
        raise DataFormatError(
            f"Could not infer the element symbol from ionization energy file name '{path.name}'."
        )

    stage_column_priority = ("ion_charge_token", "ion_stage_token", "species_name")

    records = []
    for _, row in dataframe.iterrows():
        element = source_element
        if "element" in dataframe.columns and row.get("element") not in (None, ""):
            try:
                element = parse_element_symbol(str(row.get("element")).strip())
            except ValueError:
                parsed_element = infer_element_from_filename(str(row.get("element")))
                if parsed_element is not None:
                    element = parsed_element

        stage_raw = None
        stage_source_hint = ""
        for column_name in stage_column_priority:
            value = row.get(column_name)
            if _clean_cell_value(value) is None:
                continue
            stage_raw = value
            stage_source_hint = table.original_headers.get(column_name, "")
            break

        try:
            ion_stage = parse_ionization_stage(stage_raw, stage_source_hint)
        except ValueError as exc:
            raise DataFormatError(
                f"Could not parse ionization stage in file '{path.name}' row {row.get('source_row_number')}: {exc}"
            ) from exc

        records.append(
            {
                "element": element,
                "ion_stage": ion_stage,
                "ionization_energy_eV": parse_numeric_cell(row.get("ionization_energy_eV")),
                "source_file": row.get("source_file"),
                "source_sheet": row.get("source_sheet"),
                "source_row_number": row.get("source_row_number"),
            }
        )

    logger.info(
        "Parsed ionization energy file '%s' from sheet '%s' with %d rows.",
        path.name,
        table.sheet_name,
        len(records),
    )
    return pd.DataFrame(records)


def load_cleaned_table(path: Path, logger) -> pd.DataFrame:
    dataframe = pd.read_excel(path, engine="openpyxl")
    object_columns = dataframe.select_dtypes(include=["object"]).columns
    if len(object_columns) > 0:
        dataframe.loc[:, object_columns] = dataframe.loc[:, object_columns].where(
            dataframe.loc[:, object_columns].notna(),
            None,
        )
    logger.info("Loaded cleaned table '%s' with %d rows.", path.name, len(dataframe))
    return dataframe


def sanitize_lines_table(
    *,
    element: str,
    line_table: pd.DataFrame,
    supported_stages: List[int],
    logger,
) -> pd.DataFrame:
    supported_stage_set = set(supported_stages)
    invalid_ion_mask = line_table["ion_parse_error"].notna() | line_table["ion_stage"].isna()
    unsupported_stage_mask = line_table["ion_stage"].notna() & ~line_table["ion_stage"].isin(supported_stage_set)
    missing_wavelength_mask = line_table["wavelength_observed_air_nm"].isna()
    invalid_aki_mask = line_table["Aki"].isna() | (line_table["Aki"] <= 0.0)
    invalid_upper_energy_mask = line_table["Ek_cm1"].isna()
    invalid_j_upper_mask = line_table["J_upper"].isna()
    invalid_energy_order_mask = (
        line_table["Ei_cm1"].notna()
        & line_table["Ek_cm1"].notna()
        & (line_table["Ek_cm1"] < line_table["Ei_cm1"])
    )

    dropped_invalid_rows = line_table.loc[invalid_ion_mask]
    if not dropped_invalid_rows.empty:
        logger.warning(
            "Dropping %d non-data or unparseable line rows for %s before exporting cleaned data.",
            len(dropped_invalid_rows),
            element,
        )

    dropped_unsupported_rows = line_table.loc[~invalid_ion_mask & unsupported_stage_mask]
    if not dropped_unsupported_rows.empty:
        stage_counts = (
            dropped_unsupported_rows["ion_stage"].astype(int).value_counts().sort_index().to_dict()
        )
        logger.warning(
            "Dropping %d parsed line rows for %s outside the supported stage chain %s while preparing cleaned data. "
            "Dropped stage counts = %s",
            len(dropped_unsupported_rows),
            element,
            supported_stages,
            stage_counts,
        )

    data_quality_mask = (
        missing_wavelength_mask
        | invalid_aki_mask
        | invalid_upper_energy_mask
        | invalid_j_upper_mask
        | invalid_energy_order_mask
    )
    dropped_data_quality_rows = line_table.loc[~(invalid_ion_mask | unsupported_stage_mask) & data_quality_mask]
    if not dropped_data_quality_rows.empty:
        logger.warning(
            "Dropping %d incomplete or non-physical line rows for %s before exporting cleaned data.",
            len(dropped_data_quality_rows),
            element,
        )

    cleaned = line_table.loc[~(invalid_ion_mask | unsupported_stage_mask | data_quality_mask)].copy().reset_index(drop=True)
    if cleaned.empty:
        raise DataFormatError(
            f"No cleaned line rows remain for element '{element}' after applying the supported stage chain."
        )

    logger.info(
        "Prepared cleaned line table for %s with %d rows across stages %s.",
        element,
        len(cleaned),
        supported_stages,
    )
    return cleaned


def sanitize_levels_table(
    *,
    element: str,
    ion_stage: int,
    levels_table: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    valid_mask = levels_table["J"].notna() & levels_table["level_cm1"].notna()
    dropped_invalid_rows = levels_table.loc[~valid_mask]
    if not dropped_invalid_rows.empty:
        logger.warning(
            "Dropping %d invalid level rows for %s stage %d before exporting cleaned data.",
            len(dropped_invalid_rows),
            element,
            ion_stage,
        )

    cleaned = levels_table.loc[valid_mask].copy()
    duplicate_mask = cleaned.duplicated(subset=["J", "level_cm1"], keep="first")
    dropped_duplicates = cleaned.loc[duplicate_mask]
    if not dropped_duplicates.empty:
        logger.warning(
            "Dropping %d duplicate level rows for %s stage %d before exporting cleaned data.",
            len(dropped_duplicates),
            element,
            ion_stage,
        )

    cleaned = cleaned.loc[~duplicate_mask].reset_index(drop=True)
    if cleaned.empty:
        raise DataFormatError(
            f"No cleaned levels remain for element '{element}' ion stage {ion_stage}."
        )

    logger.info(
        "Prepared cleaned levels table for %s stage %d with %d rows.",
        element,
        ion_stage,
        len(cleaned),
    )
    return cleaned


def sanitize_ionization_energy_table(
    *,
    element: str,
    ionization_energy_table: pd.DataFrame,
    supported_stages: List[int],
    logger,
) -> pd.DataFrame:
    supported_stage_set = set(supported_stages)
    filtered = ionization_energy_table.loc[
        (ionization_energy_table["element"] == element)
        & ionization_energy_table["ion_stage"].notna()
        & ionization_energy_table["ionization_energy_eV"].notna()
    ].copy()

    duplicate_mask = filtered.duplicated(subset=["ion_stage"], keep="first")
    dropped_duplicates = filtered.loc[duplicate_mask]
    if not dropped_duplicates.empty:
        logger.warning(
            "Dropping %d duplicate ionization-energy rows for %s before exporting cleaned data.",
            len(dropped_duplicates),
            element,
        )
    filtered = filtered.loc[~duplicate_mask].copy()

    dropped_unsupported_rows = filtered.loc[~filtered["ion_stage"].isin(supported_stage_set)]
    if not dropped_unsupported_rows.empty:
        logger.info(
            "Dropping %d ionization-energy rows for %s outside the supported stage chain %s "
            "while preparing cleaned data.",
            len(dropped_unsupported_rows),
            element,
            supported_stages,
        )

    cleaned = filtered.loc[filtered["ion_stage"].isin(supported_stage_set)].copy()
    cleaned = cleaned.sort_values("ion_stage").reset_index(drop=True)
    if cleaned.empty:
        raise DataFormatError(
            f"No cleaned ionization-energy rows remain for element '{element}'."
        )

    logger.info(
        "Prepared cleaned ionization-energy table for %s with %d rows covering stages %s.",
        element,
        len(cleaned),
        supported_stages,
    )
    return cleaned
