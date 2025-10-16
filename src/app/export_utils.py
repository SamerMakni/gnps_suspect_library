import os, io, re, json, math, time, hashlib, urllib.parse, datetime as dt
from pathlib import Path
from typing import Optional, Dict, Iterable, Tuple

import streamlit as st
import pandas as pd
from pyteomics import mgf
import requests
import spectrum_utils.spectrum as sus

import suspects.config as cfg  # project config

# Paths
DATA_DIR = Path(getattr(cfg, "data_dir", "data"))
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = Path(getattr(cfg, "spectra_cache_dir", DATA_DIR / "spectra_cache"))
INDEX_PARQUET = CACHE_DIR / "usi_index.parquet"
UNIQUE_PARQUET = INTERIM_DIR / "suspects_CACHE_unique.parquet"

# Limit exported entries ONLY FOR TESTING; set to None to disable
TEST_LIMIT = 50


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def get_charge(adduct: str) -> int:
    """Extract integer charge from an adduct like "[M+H]2+" or "[M-H]-"."""
    adduct_match = re.match(r"^\[.*\](\d?)([+-])$", adduct or "")
    if not adduct_match:
        return 0
    count_str, polarity = adduct_match.groups()
    charge_count = int(count_str) if count_str else 1
    return charge_count if polarity == "+" else -charge_count


def normalize_index_cols(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with columns ['usi','path'] (best effort)."""
    usi_candidates = ["usi", "USI", "library_usi", "suspect_usi", "id"]
    path_candidates = [
        "path",
        "cache_path",
        "json_path",
        "json",
        "filepath",
        "file",
        "relpath",
    ]
    usi_column = next((c for c in usi_candidates if c in dataframe.columns), None)
    path_column = next((c for c in path_candidates if c in dataframe.columns), None)
    if not usi_column or not path_column:
        raise RuntimeError("Index parquet must contain USI and JSON path columns.")

    output = dataframe.rename(columns={usi_column: "usi", path_column: "path"})[
        ["usi", "path"]
    ].copy()
    output["usi"] = output["usi"].astype(str)
    output["path"] = output["path"].astype(str)
    return output.drop_duplicates(subset=["usi"], keep="last")


def resolve_index_path(raw_path: str) -> Path:
    """Resolve stored index path to an absolute JSON file under CACHE_DIR."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    segments = path.parts
    if "spectra_cache" in segments:
        spectra_cache_index = segments.index("spectra_cache")
        relative_path = Path(*segments[spectra_cache_index + 1 :])
        return CACHE_DIR / relative_path
    return CACHE_DIR / path


def load_cached_spec(json_path: Path, fallback_usi: str) -> Optional[sus.MsmsSpectrum]:
    try:
        with open(json_path, "r") as file:
            payload = json.load(file)
        mz_values = payload.get("mz") or []
        intensity_values = payload.get("intensity") or []
        return sus.MsmsSpectrum(
            payload.get("usi", fallback_usi),
            float(payload.get("precursor_mz", 0.0)),
            int(payload.get("charge", 0)),
            mz_values,
            intensity_values,
        )
    except Exception:
        return None


def to_mgf_dict(
    spectrum: sus.MsmsSpectrum, pepmass_value: float, adduct: str
) -> Dict:
    return {
        "params": {
            "title": spectrum.identifier,
            "pepmass": float(
                pepmass_value if pepmass_value is not None else (spectrum.precursor_mz or 0.0)
            ),
            "charge": get_charge(adduct),
            "ion": adduct or "",
            "mslevel": 2,
        },
        "m/z array": spectrum.mz,
        "intensity array": spectrum.intensity,
    }



spectra_cache_memory: Dict[str, sus.MsmsSpectrum] = {}


def slug_path_for_usi(usi: str) -> Path:
    """Sharded JSON path under CACHE_DIR based on sha1(usi)."""
    sha1_hex = hashlib.sha1(usi.encode("utf-8")).hexdigest()
    return CACHE_DIR / sha1_hex[:2] / sha1_hex[2:4] / f"{sha1_hex}.json"


def save_spec_json(usi: str, spectrum: sus.MsmsSpectrum) -> Path:
    path = slug_path_for_usi(usi)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "usi": usi,
        "precursor_mz": float(spectrum.precursor_mz or 0.0),
        "charge": int(getattr(spectrum, "precursor_charge", 0) or 0),
        "mz": [float(x) for x in spectrum.mz],
        "intensity": [float(y) for y in spectrum.intensity],
    }
    with open(path, "w") as file:
        json.dump(payload, file, separators=(",", ":"))
    return path


def query_proxi_spectrum(usi: str) -> Optional[sus.MsmsSpectrum]:
    url = (
        "http://massive.ucsd.edu/ProteoSAFe/proxi/v0.1/spectra?resultType=full&usi="
        f"{urllib.parse.quote_plus(usi)}"
    )
    try:
        response = requests.get(url, timeout=None)
        response.raise_for_status()
        response_items = response.json()
        if not response_items:
            return None
        item = response_items[0]
        if "mzs" not in item:
            return None
        mz_values = [float(m) for m in item["mzs"]]
        intensity_values = [float(i) for i in item["intensities"]]
        spectrum = sus.MsmsSpectrum(usi, 0.0, 0, mz_values, intensity_values)
        spectrum.filter_intensity(0.00001)
        return spectrum
    except Exception:
        return None


def query_msv_spectrum(usi: str) -> Optional[sus.MsmsSpectrum]:
    if usi in spectra_cache_memory:
        return spectra_cache_memory[usi]

    try:
        lookup_url = f"https://massive.ucsd.edu/ProteoSAFe/QuerySpectrum?id={usi}"
        lookup_response = requests.get(lookup_url, timeout=None)
        lookup_response.raise_for_status()
        for source_file in lookup_response.json().get("row_data", []):
            if not any(
                source_file["file_descriptor"].lower().endswith(ext)
                for ext in ("mzml", "mzxml", "mgf")
            ):
                continue

            scan_id = usi.rsplit(":", 1)[-1]
            request_url = (
                "https://gnps.ucsd.edu/ProteoSAFe/DownloadResultFile?"
                "task=4f2ac74ea114401787a7e96e143bb4a1&"
                "invoke=annotatedSpectrumImageText&block=0&"
                f"file=FILE->{source_file['file_descriptor']}&"
                f"scan={scan_id}&peptide=*..*&force=false&format=JSON&uploadfile=True"
            )
            try:
                request_response = requests.get(request_url, timeout=None)
                request_response.raise_for_status()
                spectrum_data = request_response.json()
            except Exception:
                continue

            peaks_list = spectrum_data.get("peaks") or []
            if not peaks_list:
                continue

            mz_values, intensity_values = zip(*peaks_list)
            if "precursor" in spectrum_data:
                precursor_mz_value = float(spectrum_data["precursor"].get("mz", 0))
                charge_value = int(spectrum_data["precursor"].get("charge", 0))
            else:
                precursor_mz_value, charge_value = 0.0, 0

            spectrum = sus.MsmsSpectrum(
                usi, precursor_mz_value, charge_value, mz_values, intensity_values
            )
            spectrum.filter_intensity(0.00001)
            spectra_cache_memory[usi] = spectrum
            return spectrum
    except Exception:
        pass

    # fallback PROXI in case there's no MassIVE entry in the cahche
    return query_proxi_spectrum(usi)
