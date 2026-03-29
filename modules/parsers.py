from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence


@dataclass(frozen=True)
class IonIdentity:
    element: str
    ion_stage: int
    ion_label: str


@dataclass(frozen=True)
class TargetSpec:
    raw: str
    kind: Literal["element", "ion"]
    element: str
    ion_stage: Optional[int] = None
    ion_label: Optional[str] = None


WORD_STAGE_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

HEADER_ALIAS_RULES: Dict[str, Dict[str, Sequence[Sequence[str]]]] = {
    "lines": {
        "ion_label": (("ion",), ("spectrum",), ("species",), ("specie",)),
        "wavelength_observed_air_nm": (
            ("observed", "air"),
            ("obs", "air"),
            ("observed",),
            ("obs",),
        ),
        "wavelength_ritz_air_nm": (("ritz", "air"), ("ritz",)),
        "Aki": (
            ("aki",),
            ("a", "ki"),
            ("transition", "probability"),
            ("transition", "rate"),
            ("einstein", "a"),
        ),
        "Ei_cm1": (("ei",), ("lower", "cm"), ("lower", "energy")),
        "Ek_cm1": (("ek",), ("upper", "cm"), ("upper", "energy")),
        "J_lower": (("j", "i"), ("ji",), ("j", "lower"), ("lower", "j")),
        "J_upper": (("j", "k"), ("jk",), ("j", "upper"), ("upper", "j")),
    },
    "levels": {
        "J": (("j",), ("total", "angular", "momentum")),
        "level_cm1": (("level", "cm"), ("level",), ("energy", "level")),
    },
    "ionization": {
        "element": (("element",), ("atom",)),
        "ion_stage_token": (("ion",), ("charge",), ("stage",), ("spectrum",)),
        "ion_charge_token": (("ion", "charge"),),
        "species_name": (("sp", "name"), ("species", "name")),
        "ionization_energy_eV": (("ionization", "energy"), ("energy", "ev"), ("limit", "ev")),
    },
}


def normalize_header_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("\n", " ")
    text = text.replace(".", " ")
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def match_header_alias(header_text: str, dataset_kind: str) -> Optional[str]:
    normalized = normalize_header_text(header_text)
    if not normalized:
        return None
    normalized_tokens = set(normalized.split())

    for canonical_name, alias_groups in HEADER_ALIAS_RULES[dataset_kind].items():
        for alias_group in alias_groups:
            if all(token in normalized_tokens for token in alias_group):
                return canonical_name
    return None


def standardize_header_names(headers: Iterable[object], dataset_kind: str) -> Dict[int, str]:
    matched: Dict[int, str] = {}
    claimed: set[str] = set()

    for index, header in enumerate(headers):
        canonical_name = match_header_alias(str(header), dataset_kind)
        if canonical_name is None or canonical_name in claimed:
            continue
        matched[index] = canonical_name
        claimed.add(canonical_name)
    return matched


def parse_numeric_cell(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "---", "--"}:
        return None

    text = text.replace(",", "")
    text = text.replace(" ", "")
    text = text.replace("?", "")
    text = text.replace("−", "-")
    text = text.strip("[]()")

    numeric_pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    if re.fullmatch(numeric_pattern, text):
        try:
            return float(text)
        except ValueError:
            return None

    if re.search(r"[A-Za-z]", text):
        return None

    match = re.search(numeric_pattern, text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def roman_to_int(value: str) -> int:
    roman = value.strip().upper()
    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(roman):
        if character not in roman_values:
            raise ValueError(f"Invalid Roman numeral '{value}'.")
        current = roman_values[character]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    if int_to_roman(total) != roman:
        raise ValueError(f"Invalid Roman numeral '{value}'.")
    return total


def int_to_roman(value: int) -> str:
    if value <= 0:
        raise ValueError("Roman numerals require a positive integer.")

    numerals = (
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

    remaining = value
    parts = []
    for integer, numeral in numerals:
        while remaining >= integer:
            remaining -= integer
            parts.append(numeral)
    return "".join(parts)


def parse_element_symbol(value: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[A-Z][a-z]?", text):
        raise ValueError(f"Invalid chemical symbol '{value}'.")
    return text


def parse_stage_token(value: str) -> int:
    token = str(value).strip()
    if not token:
        raise ValueError("Missing ion stage token.")
    if re.fullmatch(r"\d+", token):
        stage = int(token)
        if stage <= 0:
            raise ValueError(f"Ion stage must be positive, got '{value}'.")
        return stage
    return roman_to_int(token)


def parse_ion_label(value: object, expected_element: Optional[str] = None) -> IonIdentity:
    if value is None:
        raise ValueError("Missing Ion value.")

    text = str(value).strip()
    match = re.fullmatch(r"([A-Z][a-z]?)[\s\-_]*([IVXLCDM]+|\d+)", text)
    if match is None:
        raise ValueError(f"Could not parse Ion label '{value}'. Expected formats like 'Fe I' or 'Fe II'.")

    element = parse_element_symbol(match.group(1))
    if expected_element is not None and element != expected_element:
        raise ValueError(
            f"Ion label '{value}' belongs to element '{element}', but file implies '{expected_element}'."
        )

    ion_stage = parse_stage_token(match.group(2))
    return IonIdentity(
        element=element,
        ion_stage=ion_stage,
        ion_label=f"{element} {int_to_roman(ion_stage)}",
    )


def parse_target_string(value: str) -> TargetSpec:
    raw = value.strip()
    if not raw:
        raise ValueError("Target entries must not be empty.")

    if re.fullmatch(r"[A-Z][a-z]?", raw):
        element = parse_element_symbol(raw)
        return TargetSpec(raw=element, kind="element", element=element)

    ion_identity = parse_ion_label(raw)
    return TargetSpec(
        raw=ion_identity.ion_label,
        kind="ion",
        element=ion_identity.element,
        ion_stage=ion_identity.ion_stage,
        ion_label=ion_identity.ion_label,
    )


def parse_j_value(value: object) -> float:
    if value is None:
        raise ValueError("Missing J value.")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        raise ValueError("Missing J value.")

    try:
        if "/" in text:
            return float(Fraction(text.replace(" ", "")))
        return float(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Could not parse J value '{value}'.") from exc


def infer_element_from_filename(filename: str) -> Optional[str]:
    stem = Path(filename).stem
    match = re.match(r"([A-Z][a-z]?)", stem)
    if match is None:
        return None
    return match.group(1)


def infer_stage_from_filename(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", stem) if token]

    for token in tokens:
        token_lower = token.lower()
        if token_lower in {"levels", "level", "energy", "energies", "ionization", "lines", "line"}:
            continue
        if token_lower in WORD_STAGE_MAP:
            return WORD_STAGE_MAP[token_lower]
        if re.fullmatch(r"\d+", token):
            stage = int(token)
            if stage > 0:
                return stage
        if re.fullmatch(r"[IVXLCDM]+", token.upper()):
            return roman_to_int(token.upper())
    return None


def parse_ionization_stage(value: object, source_hint: str) -> Optional[int]:
    if value is None:
        return None

    hint_normalized = normalize_header_text(source_hint)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        if numeric_value != numeric_value:
            return None
        if numeric_value.is_integer():
            integer_value = int(numeric_value)
            if "charge" in hint_normalized:
                if integer_value < 0:
                    raise ValueError(f"Negative ion charge is invalid: '{value}'.")
                return integer_value + 1
            if integer_value == 0:
                return 1
            if integer_value > 0:
                return integer_value

    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"[A-Z][a-z]?[\s\-_]*([IVXLCDM]+|\d+)", text):
        return parse_ion_label(text).ion_stage
    if re.fullmatch(r"[IVXLCDM]+", text.upper()):
        return roman_to_int(text.upper())
    if re.fullmatch(r"[+-]?\d+\+?", text):
        integer_value = int(re.search(r"[+-]?\d+", text).group(0))
        if "charge" in hint_normalized:
            if integer_value < 0:
                raise ValueError(f"Negative ion charge is invalid: '{value}'.")
            return integer_value + 1
        if integer_value == 0:
            return 1
        if integer_value > 0:
            return integer_value
    return None
