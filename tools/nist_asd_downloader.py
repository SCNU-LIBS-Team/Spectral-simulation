from __future__ import annotations

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ASD_BASE_URL = "https://physics.nist.gov"
LINES_ENDPOINT = f"{ASD_BASE_URL}/cgi-bin/ASD/lines1.pl"
LEVELS_ENDPOINT = f"{ASD_BASE_URL}/cgi-bin/ASD/energy1.pl"
IONIZATION_ENDPOINT = f"{ASD_BASE_URL}/cgi-bin/ASD/ie.pl"
USER_AGENT = "SpectralSimulation-NIST-ASD-Downloader/1.0"

ROMAN_NUMERALS = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)

PERIODIC_TABLE = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

NO_DATA_PATTERNS = (
    "no lines are available",
    "no levels are available",
    "no energy levels are available",
    "no ground states and ionization energies are available",
    "no data are available",
    "no spectra are selected",
)


@dataclass(frozen=True)
class DownloaderConfig:
    output_root: Path
    elements: list[str]
    min_stage: int
    max_stage: int
    line_min_nm: float | None
    line_max_nm: float | None
    line_chunk_nm: float | None
    timeout_seconds: float
    sleep_seconds: float
    overwrite: bool
    save_raw: bool


@dataclass(frozen=True)
class RequestResult:
    text: str
    content_type: str
    url: str


def int_to_roman(value: int) -> str:
    if value <= 0:
        raise ValueError("Roman numerals require a positive integer.")

    remaining = value
    parts: list[str] = []
    for integer, numeral in ROMAN_NUMERALS:
        while remaining >= integer:
            remaining -= integer
            parts.append(numeral)
    return "".join(parts)


def parse_element_list(raw_elements: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    seen: set[str] = set()

    for raw_value in raw_elements:
        for token in re.split(r"[\s,;]+", raw_value.strip()):
            if not token:
                continue
            if not re.fullmatch(r"[A-Z][a-z]?", token):
                raise ValueError(f"Invalid chemical symbol '{token}'.")
            if token not in seen:
                parsed.append(token)
                seen.add(token)
    return parsed


def parse_arguments() -> DownloaderConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Download NIST ASD lines, levels, and ionization-energy tables into a new local folder "
            "without touching the project's existing data files."
        )
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        help="Element symbols, for example: --elements Fe Cu Al or --elements Fe,Cu,Al",
    )
    parser.add_argument(
        "--all-elements",
        action="store_true",
        help="Download for the full periodic table instead of passing --elements.",
    )
    parser.add_argument(
        "--min-stage",
        type=int,
        default=1,
        help="Lowest ion stage to request for lines/levels. Default: 1",
    )
    parser.add_argument(
        "--max-stage",
        type=int,
        default=3,
        help="Highest ion stage to request for lines/levels. Default: 3",
    )
    parser.add_argument(
        "--line-min-nm",
        type=float,
        default=200.0,
        help="Lower wavelength bound for line downloads in nm. Default: 200",
    )
    parser.add_argument(
        "--line-max-nm",
        type=float,
        default=900.0,
        help="Upper wavelength bound for line downloads in nm. Default: 900",
    )
    parser.add_argument(
        "--line-chunk-nm",
        type=float,
        default=100.0,
        help="Chunk size for line downloads in nm. Default: 100",
    )
    parser.add_argument(
        "--all-wavelengths",
        action="store_true",
        help="Disable wavelength limits and try to fetch each ion stage in a single lines request.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("downloaded_nist_data"),
        help="Root directory for downloaded files. Default: downloaded_nist_data",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Pause between requests in seconds. Default: 0.5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-downloaded files.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save the raw CSV text returned by NIST under _raw_csv/.",
    )
    args = parser.parse_args()

    if args.all_elements:
        elements = PERIODIC_TABLE[:]
    elif args.elements:
        elements = parse_element_list(args.elements)
    else:
        parser.error("Pass --elements or use --all-elements.")

    if args.min_stage <= 0 or args.max_stage <= 0:
        parser.error("--min-stage and --max-stage must be positive integers.")
    if args.min_stage > args.max_stage:
        parser.error("--min-stage must be <= --max-stage.")

    line_min_nm: float | None = args.line_min_nm
    line_max_nm: float | None = args.line_max_nm
    line_chunk_nm: float | None = args.line_chunk_nm
    if args.all_wavelengths:
        line_min_nm = None
        line_max_nm = None
        line_chunk_nm = None
    else:
        if line_min_nm is None or line_max_nm is None:
            parser.error("When --all-wavelengths is not used, both --line-min-nm and --line-max-nm are required.")
        if line_min_nm >= line_max_nm:
            parser.error("--line-min-nm must be smaller than --line-max-nm.")
        if line_chunk_nm is not None and line_chunk_nm <= 0.0:
            parser.error("--line-chunk-nm must be positive.")

    return DownloaderConfig(
        output_root=args.output_root.resolve(),
        elements=elements,
        min_stage=args.min_stage,
        max_stage=args.max_stage,
        line_min_nm=line_min_nm,
        line_max_nm=line_max_nm,
        line_chunk_nm=line_chunk_nm,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
        overwrite=args.overwrite,
        save_raw=args.save_raw,
    )


def ensure_directories(config: DownloaderConfig) -> dict[str, Path]:
    directories = {
        "root": config.output_root,
        "lines": config.output_root / "Lines_data",
        "levels": config.output_root / "Levels_data",
        "ionization": config.output_root / "Ionization_Energies_data",
        "raw_root": config.output_root / "_raw_csv",
        "raw_lines": config.output_root / "_raw_csv" / "lines",
        "raw_levels": config.output_root / "_raw_csv" / "levels",
        "raw_ionization": config.output_root / "_raw_csv" / "ionization",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def request_text(url: str, params: dict[str, object], timeout_seconds: float) -> RequestResult:
    encoded_query = urlencode(params, doseq=True)
    full_url = f"{url}?{encoded_query}"
    request = Request(full_url, headers={"User-Agent": USER_AGENT})

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            content_type = response.headers.get("Content-Type", "")
            return RequestResult(text=payload.decode(charset, errors="replace"), content_type=content_type, url=full_url)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if is_no_data_response(body):
            return RequestResult(text=body, content_type=exc.headers.get("Content-Type", ""), url=full_url)
        raise RuntimeError(f"NIST request failed with HTTP {exc.code}: {full_url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach NIST ASD: {exc}") from exc


def is_no_data_response(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in NO_DATA_PATTERNS)


def assert_csv_like_response(result: RequestResult) -> None:
    lowered = result.text.lstrip().lower()
    if "<html" in lowered or "<!doctype html" in lowered:
        snippet = result.text[:300].replace("\n", " ")
        raise RuntimeError(f"NIST returned HTML instead of CSV/text for URL: {result.url}\nSnippet: {snippet}")


def clean_csv_cell(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    text = text.replace('""', '"')
    text = text.strip()
    if not text:
        return pd.NA
    return text


def parse_csv_table(csv_text: str) -> pd.DataFrame:
    dataframe = pd.read_csv(StringIO(csv_text), dtype=str, keep_default_na=False)
    dataframe.columns = [str(clean_csv_cell(column) or "").strip() for column in dataframe.columns]

    drop_columns = []
    for column in dataframe.columns:
        if not column or column.lower().startswith("unnamed"):
            drop_columns.append(column)
    if drop_columns:
        dataframe = dataframe.drop(columns=drop_columns)

    dataframe = dataframe.apply(lambda column: column.map(clean_csv_cell))
    dataframe = dataframe.dropna(axis=1, how="all")
    dataframe = dataframe.dropna(axis=0, how="all")
    return dataframe.reset_index(drop=True)


def levels_query_params(element: str, stage: int) -> dict[str, object]:
    return {
        "spectrum": f"{element} {int_to_roman(stage)}",
        "units": 0,
        "format": 2,
        "output": 0,
        "page_size": 15,
        "multiplet_ordered": 1,
        "conf_out": 1,
        "term_out": 1,
        "level_out": 1,
        "unc_out": 1,
        "j_out": 1,
        "lande_out": 1,
        "perc_out": 1,
        "biblio": 1,
    }


def ionization_query_params(element: str) -> dict[str, object]:
    return {
        "spectra": element,
        "units": 1,
        "format": 2,
        "order": 0,
        "at_num_out": 1,
        "sp_name_out": 1,
        "ion_charge_out": 1,
        "el_name_out": 1,
        "seq_out": 1,
        "shells_out": 1,
        "conf_out": 1,
        "level_out": 1,
        "ion_conf_out": 1,
        "e_out": 0,
        "unc_out": 1,
        "biblio": 1,
        "remove_js": "on",
    }


def lines_query_params(element: str, stage: int, low_w_nm: float | None, upp_w_nm: float | None) -> dict[str, object]:
    return {
        "spectra": f"{element} {int_to_roman(stage)}",
        "limits_type": 0,
        "low_w": "" if low_w_nm is None else f"{low_w_nm:.6f}",
        "upp_w": "" if upp_w_nm is None else f"{upp_w_nm:.6f}",
        "unit": 1,
        "de": 0,
        "format": 2,
        "line_out": 0,
        "remove_js": "on",
        "en_unit": 0,
        "output": 0,
        "page_size": 15,
        "order_out": 0,
        "show_av": 2,
        "output_type": 0,
        "show_obs_wl": 1,
        "show_calc_wl": 1,
        "unc_out": 1,
        "A_out": 0,
        "intens_out": "on",
        "allowed_out": 1,
        "forbid_out": 1,
        "bibrefs": 1,
        "conf_out": "on",
        "term_out": "on",
        "enrg_out": "on",
        "J_out": "on",
    }


def line_windows(config: DownloaderConfig) -> list[tuple[float | None, float | None]]:
    if config.line_min_nm is None or config.line_max_nm is None:
        return [(None, None)]
    if config.line_chunk_nm is None:
        return [(config.line_min_nm, config.line_max_nm)]

    windows: list[tuple[float, float]] = []
    start = config.line_min_nm
    while start < config.line_max_nm:
        stop = min(start + config.line_chunk_nm, config.line_max_nm)
        windows.append((start, stop))
        start = stop
    return windows


def save_raw_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_dataframe(path: Path, dataframe: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_excel(path, index=False)


def download_levels_table(
    *,
    element: str,
    stage: int,
    config: DownloaderConfig,
    raw_dir: Path,
) -> tuple[str, int, Path | None]:
    output_path = config.output_root / "Levels_data" / f"{element}_{int_to_roman(stage)}_Levels.xlsx"
    if output_path.exists() and not config.overwrite:
        existing_rows = len(pd.read_excel(output_path))
        return "skipped_existing", existing_rows, output_path

    result = request_text(LEVELS_ENDPOINT, levels_query_params(element, stage), config.timeout_seconds)
    if is_no_data_response(result.text):
        return "no_data", 0, None

    assert_csv_like_response(result)
    dataframe = parse_csv_table(result.text)
    if dataframe.empty:
        return "no_data", 0, None

    if config.save_raw:
        save_raw_text(raw_dir / f"{element}_{int_to_roman(stage)}_Levels.csv", result.text, config.overwrite)
    save_dataframe(output_path, dataframe, config.overwrite)
    return "downloaded", len(dataframe), output_path


def download_ionization_table(
    *,
    element: str,
    config: DownloaderConfig,
    raw_dir: Path,
) -> tuple[str, int, Path | None]:
    output_path = config.output_root / "Ionization_Energies_data" / f"{element}_energy.xlsx"
    if output_path.exists() and not config.overwrite:
        existing_rows = len(pd.read_excel(output_path))
        return "skipped_existing", existing_rows, output_path

    result = request_text(IONIZATION_ENDPOINT, ionization_query_params(element), config.timeout_seconds)
    if is_no_data_response(result.text):
        return "no_data", 0, None

    assert_csv_like_response(result)
    dataframe = parse_csv_table(result.text)
    if dataframe.empty:
        return "no_data", 0, None

    if config.save_raw:
        save_raw_text(raw_dir / f"{element}_energy.csv", result.text, config.overwrite)
    save_dataframe(output_path, dataframe, config.overwrite)
    return "downloaded", len(dataframe), output_path


def download_lines_table(
    *,
    element: str,
    config: DownloaderConfig,
    raw_dir: Path,
) -> tuple[str, int, Path | None]:
    output_path = config.output_root / "Lines_data" / f"{element}_lines.xlsx"
    if output_path.exists() and not config.overwrite:
        existing_rows = len(pd.read_excel(output_path))
        return "skipped_existing", existing_rows, output_path

    collected_frames: list[pd.DataFrame] = []
    windows = line_windows(config)

    for stage in range(config.min_stage, config.max_stage + 1):
        ion_label = f"{element} {int_to_roman(stage)}"
        stage_frames: list[pd.DataFrame] = []

        for window_index, (low_w_nm, upp_w_nm) in enumerate(windows, start=1):
            result = request_text(
                LINES_ENDPOINT,
                lines_query_params(element, stage, low_w_nm, upp_w_nm),
                config.timeout_seconds,
            )
            if is_no_data_response(result.text):
                continue

            assert_csv_like_response(result)
            dataframe = parse_csv_table(result.text)
            if dataframe.empty:
                continue

            dataframe.insert(0, "Spectrum", ion_label)
            dataframe.insert(1, "Spectrum Query", ion_label)
            if low_w_nm is not None and upp_w_nm is not None:
                dataframe.insert(2, "Requested Low (nm)", f"{low_w_nm:.6f}")
                dataframe.insert(3, "Requested Upp (nm)", f"{upp_w_nm:.6f}")

            if config.save_raw:
                if low_w_nm is None or upp_w_nm is None:
                    raw_name = f"{element}_{int_to_roman(stage)}_lines.csv"
                else:
                    raw_name = (
                        f"{element}_{int_to_roman(stage)}_lines_"
                        f"{window_index:03d}_{low_w_nm:.1f}_{upp_w_nm:.1f}nm.csv"
                    )
                save_raw_text(raw_dir / raw_name, result.text, config.overwrite)

            stage_frames.append(dataframe)
            time.sleep(config.sleep_seconds)

        if not stage_frames:
            print(f"[lines] {ion_label}: no data returned")
            continue

        stage_table = pd.concat(stage_frames, ignore_index=True)
        stage_table = stage_table.drop_duplicates().reset_index(drop=True)
        collected_frames.append(stage_table)
        print(f"[lines] {ion_label}: {len(stage_table)} rows")

    if not collected_frames:
        return "no_data", 0, None

    combined = pd.concat(collected_frames, ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)
    save_dataframe(output_path, combined, config.overwrite)
    return "downloaded", len(combined), output_path


def manifest_row(
    *,
    dataset: str,
    element: str,
    stage: int | None,
    status: str,
    rows: int,
    path: Path | None,
    note: str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "element": element,
        "stage": "" if stage is None else stage,
        "status": status,
        "rows": rows,
        "path": "" if path is None else str(path),
        "note": note,
    }


def run() -> int:
    try:
        config = parse_arguments()
    except Exception as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    ensure_directories(config)
    manifest: list[dict[str, object]] = []
    failure_count = 0

    print("NIST ASD downloader")
    print(f"Output root: {config.output_root}")
    print(f"Elements: {', '.join(config.elements)}")
    print(f"Stages: {config.min_stage}..{config.max_stage}")
    if config.line_min_nm is None or config.line_max_nm is None:
        print("Lines wavelength window: all wavelengths")
    else:
        print(
            "Lines wavelength window: "
            f"{config.line_min_nm:.3f}..{config.line_max_nm:.3f} nm "
            f"(chunk {config.line_chunk_nm:.3f} nm)"
        )

    raw_levels_dir = config.output_root / "_raw_csv" / "levels"
    raw_lines_dir = config.output_root / "_raw_csv" / "lines"
    raw_ionization_dir = config.output_root / "_raw_csv" / "ionization"

    for element in config.elements:
        print(f"\n=== {element} ===")

        try:
            status, rows, path = download_ionization_table(
                element=element,
                config=config,
                raw_dir=raw_ionization_dir,
            )
            print(f"[ionization] {element}: {status} ({rows} rows)")
            manifest.append(
                manifest_row(
                    dataset="ionization",
                    element=element,
                    stage=None,
                    status=status,
                    rows=rows,
                    path=path,
                )
            )
        except Exception as exc:
            failure_count += 1
            print(f"[ionization] {element}: error -> {exc}")
            manifest.append(
                manifest_row(
                    dataset="ionization",
                    element=element,
                    stage=None,
                    status="error",
                    rows=0,
                    path=None,
                    note=str(exc),
                )
            )

        for stage in range(config.min_stage, config.max_stage + 1):
            try:
                status, rows, path = download_levels_table(
                    element=element,
                    stage=stage,
                    config=config,
                    raw_dir=raw_levels_dir,
                )
                print(f"[levels] {element} {int_to_roman(stage)}: {status} ({rows} rows)")
                manifest.append(
                    manifest_row(
                        dataset="levels",
                        element=element,
                        stage=stage,
                        status=status,
                        rows=rows,
                        path=path,
                    )
                )
            except Exception as exc:
                failure_count += 1
                print(f"[levels] {element} {int_to_roman(stage)}: error -> {exc}")
                manifest.append(
                    manifest_row(
                        dataset="levels",
                        element=element,
                        stage=stage,
                        status="error",
                        rows=0,
                        path=None,
                        note=str(exc),
                    )
                )
            time.sleep(config.sleep_seconds)

        try:
            status, rows, path = download_lines_table(
                element=element,
                config=config,
                raw_dir=raw_lines_dir,
            )
            print(f"[lines] {element}: {status} ({rows} rows)")
            manifest.append(
                manifest_row(
                    dataset="lines",
                    element=element,
                    stage=None,
                    status=status,
                    rows=rows,
                    path=path,
                )
            )
        except Exception as exc:
            failure_count += 1
            print(f"[lines] {element}: error -> {exc}")
            manifest.append(
                manifest_row(
                    dataset="lines",
                    element=element,
                    stage=None,
                    status="error",
                    rows=0,
                    path=None,
                    note=str(exc),
                )
            )
        time.sleep(config.sleep_seconds)

    manifest_path = config.output_root / "download_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"\nManifest written to: {manifest_path}")

    if failure_count:
        print(f"Completed with {failure_count} failure(s). See the manifest for details.", file=sys.stderr)
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
