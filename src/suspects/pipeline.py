import glob
import logging
import math
import operator
import os
import re

from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
import joblib
import numpy as np
import pandas as pd
import pyteomics.mass.unimod as unimod
import scipy.signal as ssignal
import tqdm

from . import config
from .loader import (
    generate_suspects_global as _generate_suspects_global,
    read_ids as _read_ids,
    load_living_data as _generate_suspects_per_dataset,
)

def _report(report_cb, msg: str, progress: float | None = None):
    if report_cb:
        try:
            report_cb(msg, progress)
        except Exception:
            # never crash the pipeline because of UI
            pass

logger = logging.getLogger("suspect_library")


# def generate_suspects(filenames_tsv: pd.DataFrame) -> None:
#     """
#     Generate suspects from the GNPS living data results.

#     Suspect (unfiltered and filtered, unique) metadata is exported to csv files
#     in the data directory.

#     Settings for the suspect generation are taken from the config file.
#     """
#     # if config.filename_ids is not None:
#     #     task_id = re.match(
#     #         r"MOLECULAR-LIBRARYSEARCH-V2-([a-z0-9]{8})-"
#     #         r"view_all_annotations_DB-main.tsv",
#     #         os.path.basename(config.filename_ids),
#     #     ).group(1)
#     # else:
#     #     task_id = re.search(
#     #         r"MSV000084314/updates/\d{4}-\d{2}-\d{2}_.+_([a-z0-9]{8})/other",
#     #         config.living_data_dir,
#     #     ).group(1)
#     task_id = "TEST"  # since we are not using the original GNPS files
#     suspects_dir = os.path.join(config.data_dir, "interim")

#     # Get the clustering data per individual dataset.
#     ids, pairs, clusters = _generate_suspects_per_dataset(
#         config.living_data_dir
#     )
#     logger.info(
#         "%d spectrum annotations, %d spectrum pairs, %d clusters retrieved "
#         "from living data results for individual datasets",
#         *map(lambda i: sum(map(len, i)), (ids, pairs, clusters)),
#     )
#     # Get the clustering data from the global analysis.
#     ids_g, pairs_g, clusters_g = _generate_suspects_global(
#         config.global_network_dir, config.global_network_task_id
#     )
#     logger.info(
#         "%d spectrum annotations, %d spectrum pairs, %d clusters retrieved "
#         "from the global molecular network",
#         *map(len, (ids_g, pairs_g, clusters_g)),
#     )
#     # Merge the clustering data from both sources.
#     ids.append(ids_g)
#     pairs.append(pairs_g)
#     clusters.append(clusters_g)
#     if config.filename_ids is not None:
#         print(type(filenames_tsv))
#         extra_ids = _read_ids(filenames_tsv)
#         ids.append(extra_ids)
#         if extra_ids:
#             logger.info(
#                 "%d additional spectrum annotations from external library "
#                 "searching included",
#                 len(extra_ids)
#             )
#             library_usis_to_include = set(extra_ids["LibraryUsi"])
#         else:
#             library_usis_to_include = None
#     else:
#         library_usis_to_include = None
#     ids = pd.concat(ids, ignore_index=True, copy=False)
#     pairs = pd.concat(pairs, ignore_index=True, copy=False)
#     clusters = pd.concat(clusters, ignore_index=True, copy=False)
#     logger.info(
#         "%d spectrum annotations, %d spectrum pairs, %d clusters retained "
#         "before filtering",
#         *map(len, (ids, pairs, clusters)),
#     )
#     # Filter based on the defined acceptance criteria.
#     ids = _filter_ids(ids, config.max_ppm, config.min_shared_peaks)
#     pairs = _filter_pairs(pairs, config.min_cosine)
#     clusters = _filter_clusters(clusters)
#     logger.info(
#         "%d spectrum annotations, %d spectrum pairs, %d clusters retained "
#         "after filtering",
#         *map(len, (ids, pairs, clusters)),
#     )

#     # Generate suspects from the full clustering data.
#     suspects_unfiltered = _generate_suspects(ids, pairs, clusters)
#     print("Δm describe:\n", suspects_unfiltered["DeltaMass"].describe())
#     for thr in [0.5, 0.2, 0.1, 0.05, 0.01]:
#         cnt = (suspects_unfiltered["DeltaMass"].abs() > thr).sum()
#         print(f"count |Δm| > {thr}: {cnt}")
#     suspects_unfiltered.to_parquet(
#         os.path.join(suspects_dir, f"suspects_{task_id}_unfiltered.parquet"),
#         index=False,
#     )
#     logger.info(
#         "%d candidate unfiltered suspects generated", len(suspects_unfiltered)
#     )
#     # Ignore suspects without a mass shift.
#     suspects_grouped = suspects_unfiltered[
#         suspects_unfiltered["DeltaMass"].abs() > config.min_delta_mz
#     ].copy()
#     # Group and assign suspects by observed mass shift.
#     suspects_grouped = _group_mass_shifts(
#         suspects_grouped,
#         _get_mass_shift_annotations(
#             config.mass_shift_annotation_url, getattr(config, "unimod_file", None)
#         ),
#         config.interval_width,
#         config.bin_width,
#         config.peak_height,
#         config.max_dist,
#     )
#     # Ignore ungrouped suspects.
#     suspects_grouped = suspects_grouped.dropna(subset=["GroupDeltaMass"])
#     # (Optionally) filter by the supplementary identifications.
#     if library_usis_to_include is not None:
#         suspects_grouped = suspects_grouped[
#             suspects_grouped["LibraryUsi"].isin(library_usis_to_include)
#         ]
#     suspects_grouped.to_parquet(
#         os.path.join(suspects_dir, f"suspects_{task_id}_grouped.parquet"),
#         index=False,
#     )
#     logger.info(
#         "%d (non-unique) suspects with non-zero mass differences collected",
#         len(suspects_grouped),
#     )

#     # 1. Only use the top suspect (by cosine score) per combination of library
#     #    spectrum and grouped mass shift.
#     # 2. Avoid repeated occurrences of the same suspect with different adducts.
#     suspects_unique = (
#         suspects_grouped.sort_values("Cosine", ascending=False)
#         .drop_duplicates(["CompoundName", "Adduct", "GroupDeltaMass"])
#         .sort_values("Adduct", key=_get_adduct_n_elements)
#         .drop_duplicates(["CompoundName", "SuspectUsi"])
#         .sort_values(["CompoundName", "Adduct", "GroupDeltaMass"])
#     )
#     suspects_unique.to_parquet(
#         os.path.join(suspects_dir, f"suspects_{task_id}_unique.parquet"),
#         index=False,
#     )
#     logger.info(
#         "%d unique suspects retained after duplicate removal and filtering",
#         len(suspects_unique),
#     )

def generate_suspects(filenames_tsv: pd.DataFrame, report_cb=None) -> str:
    """
    Same pipeline as before. CLI logs unchanged.
    If report_cb is provided, send step-by-step messages for Streamlit.
    """
    task_id = "TEST"
    suspects_dir = os.path.join(config.data_dir, "interim")

    # 0) starting
    _report(report_cb, "Starting pipeline…", 0.02)

    # A) load living data
    _report(report_cb, "Loading living data…", 0.08)
    ids, pairs, clusters = _generate_suspects_per_dataset(config.living_data_dir)
    msg = (f"{sum(map(len, ids))} spectrum annotations, "
           f"{sum(map(len, pairs))} spectrum pairs, "
           f"{sum(map(len, clusters))} clusters retrieved from living data results for individual datasets")
    logger.info(msg)
    _report(report_cb, msg, 0.12)

    # B) load global network
    _report(report_cb, "Loading global network…", 0.18)
    ids_g, pairs_g, clusters_g = _generate_suspects_global(
        config.global_network_dir, config.global_network_task_id
    )
    msg = (f"{len(ids_g)} spectrum annotations, "
           f"{len(pairs_g)} spectrum pairs, "
           f"{len(clusters_g)} clusters retrieved from the global molecular network")
    logger.info(msg)
    _report(report_cb, msg, 0.24)

    # C) merge
    _report(report_cb, "Merging inputs…", 0.30)
    ids.append(ids_g); pairs.append(pairs_g); clusters.append(clusters_g)

    library_usis_to_include = None
    if config.filename_ids is not None:
        _report(report_cb, "Reading extra annotations…", 0.34)
        extra_ids = _read_ids(filenames_tsv)
        ids.append(extra_ids)
        if extra_ids is not None and len(extra_ids):
            logger.info("%d additional spectrum annotations from external library searching included", len(extra_ids))
            _report(report_cb, f"{len(extra_ids)} additional annotations included", 0.38)
            library_usis_to_include = set(extra_ids["LibraryUsi"])

    ids = pd.concat(ids, ignore_index=True, copy=False)
    pairs = pd.concat(pairs, ignore_index=True, copy=False)
    clusters = pd.concat(clusters, ignore_index=True, copy=False)
    msg = (f"{len(ids)} spectrum annotations, {len(pairs)} spectrum pairs, "
           f"{len(clusters)} clusters retained before filtering")
    logger.info(msg)
    _report(report_cb, msg, 0.42)

    # D) filter
    _report(report_cb, "Filtering IDs, pairs, and clusters…", 0.48)
    ids = _filter_ids(ids, config.max_ppm, config.min_shared_peaks)
    pairs = _filter_pairs(pairs, config.min_cosine)
    clusters = _filter_clusters(clusters)
    msg = (f"{len(ids)} spectrum annotations, {len(pairs)} spectrum pairs, "
           f"{len(clusters)} clusters retained after filtering")
    logger.info(msg)
    _report(report_cb, msg, 0.56)

    # E) generate unfiltered
    _report(report_cb, "Generating unfiltered suspects…", 0.62)
    suspects_unfiltered = _generate_suspects(ids, pairs, clusters)
    logger.info("%d candidate unfiltered suspects generated", len(suspects_unfiltered))
    _report(report_cb, f"{len(suspects_unfiltered)} candidate unfiltered suspects", 0.66)

    # optional Δm stats – keep to logger; send a compact line to UI
    dm_desc = suspects_unfiltered["DeltaMass"].describe()
    logger.info("Δm describe:\n%s", dm_desc.to_string())
    _report(report_cb, "Computed Δm statistics", 0.68)

    # F) group mass shifts
    _report(report_cb, "Grouping mass shifts and assigning rationales…", 0.74)
    suspects_grouped = suspects_unfiltered[
        suspects_unfiltered["DeltaMass"].abs() > config.min_delta_mz
    ].copy()
    suspects_grouped = _group_mass_shifts(
        suspects_grouped,
        _get_mass_shift_annotations(
            config.mass_shift_annotation_url, getattr(config, "unimod_file", None)
        ),
        config.interval_width,
        config.bin_width,
        config.peak_height,
        config.max_dist,
    )
    suspects_grouped = suspects_grouped.dropna(subset=["GroupDeltaMass"])

    if library_usis_to_include is not None:
        _report(report_cb, "Applying extra-annotation whitelist…", 0.78)
        suspects_grouped = suspects_grouped[
            suspects_grouped["LibraryUsi"].isin(library_usis_to_include)
        ]

    # save unfiltered + grouped
    _report(report_cb, "Writing unfiltered & grouped parquet…", 0.82)
    os.makedirs(suspects_dir, exist_ok=True)
    suspects_unfiltered.to_parquet(
        os.path.join(suspects_dir, f"suspects_{task_id}_unfiltered.parquet"),
        index=False,
    )
    suspects_grouped.to_parquet(
        os.path.join(suspects_dir, f"suspects_{task_id}_grouped.parquet"),
        index=False,
    )
    logger.info("%d (non-unique) suspects with non-zero mass differences collected", len(suspects_grouped))
    _report(report_cb, f"{len(suspects_grouped)} grouped suspects saved", 0.86)

    # G) dedupe → unique
    _report(report_cb, "Selecting unique suspects…", 0.90)
    suspects_unique = (
        suspects_grouped.sort_values("Cosine", ascending=False)
        .drop_duplicates(["CompoundName", "Adduct", "GroupDeltaMass"])
        .sort_values("Adduct", key=_get_adduct_n_elements)
        .drop_duplicates(["CompoundName", "SuspectUsi"])
        .sort_values(["CompoundName", "Adduct", "GroupDeltaMass"])
    )
    suspects_unique.to_parquet(
        os.path.join(suspects_dir, f"suspects_{task_id}_unique.parquet"),
        index=False,
    )
    logger.info("%d unique suspects retained after duplicate removal and filtering", len(suspects_unique))
    _report(report_cb, f"{len(suspects_unique)} unique suspects saved", 0.96)
    
    final_message = (f"Generation complete: "
                     f"{len(suspects_unfiltered)} unfiltered, "
                     f"{len(suspects_grouped)} grouped, "
                     f"{len(suspects_unique)} unique suspects. "
                     f"From {len(clusters)} clusters.")
    _report(report_cb, final_message, 1.0)
    logger.info(final_message)
    return final_message


def _filter_ids(
    ids: pd.DataFrame, max_ppm: float, min_shared_peaks: int
) -> pd.DataFrame:
    """
    Filter high-quality identifications according to the given maximum ppm
    deviation and minimum number of shared peaks.

    Clean the metadata (instrument, ion source, ion mode, adduct).

    Arguments
    ---------
    ids : pd.DataFrame
        The tabular identifications retrieved from GNPS.
    max_ppm : float
        The maximum ppm deviation.
    min_shared_peaks : int
        The minimum number of shared peaks.

    Returns
    -------
    pd.DataFrame
        The identifications retained after filtering.
    """
    # Clean the metadata.
    ids["Instrument"] = ids["Instrument"].replace(
        {
            # Hybrid FT.
            "ESI-QFT": "Hybrid FT",
            "Hybrid Ft": "Hybrid FT",
            "IT-FT/ion trap with FTMS": "Hybrid FT",
            "LC-ESI-ITFT": "Hybrid FT",
            "LC-ESI-QFT": "Hybrid FT",
            "LTQ-FT-ICR": "Hybrid FT",
            # Ion Trap.
            "CID; Velos": "Ion Trap",
            "IT/ion trap": "Ion Trap",
            "Ger": "Ion Trap",
            "LCQ": "Ion Trap",
            "QqIT": "Ion Trap",
            # qToF.
            " impact HD": "qTof",
            "ESI-QTOF": "qTof",
            "LC-ESI-QTOF": "qTof",
            "LC-Q-TOF/MS": "qTof",
            "Maxis HD qTOF": "qTof",
            "qToF": "qTof",
            "Maxis II HD Q-TOF Bruker": "qTof",
            "Q-TOF": "qTof",
            "qTOF": "qTof",
            # QQQ.
            "BEqQ/magnetic and electric sectors with quadrupole": "QQQ",
            "LC-APPI-QQ": "QQQ",
            "LC-ESI-QQ": "QQQ",
            "QqQ": "QQQ",
            "Quattro_QQQ:10eV": "QQQ",
            "Quattro_QQQ:25eV": "QQQ",
            "QqQ/triple quadrupole": "QQQ",
            # Orbitrap.
            "HCD": "Orbitrap",
            "HCD; Lumos": "Orbitrap",
            "HCD; Velos": "Orbitrap",
            "Q-Exactive Plus": "Orbitrap",
            "Q-Exactive Plus Orbitrap Res 70k": "Orbitrap",
            "Q-Exactive Plus Orbitrap Res 14k": "Orbitrap",
        }
    ).astype("category")
    ids["IonSource"] = ids["IonSource"].replace(
        {
            "CI": "APCI",
            "CI (MeOH)": "APCI",
            "DI-ESI": "ESI",
            "ESI/APCI": "APCI",
            "LC-APCI": "APCI",
            "in source ESI": "ESI",
            "LC-ESI-QFT": "LC-ESI",
            "LC-ESIMS": "LC-ESI",
            " ": "ESI",
            "Positive": "ESI",
        }
    ).astype("category")
    ids["IonMode"] = (
        ids["IonMode"].str.strip().str.capitalize().str.split("-", n=1).str[0]
    ).astype("category")
    ids["Adduct"] = ids["Adduct"].astype(str).apply(_clean_adduct)

    return ids[
        (ids["MzErrorPpm"].abs() <= max_ppm)
        & (ids["SharedPeaks"] >= min_shared_peaks)
    ].dropna(subset=["Instrument", "IonSource", "IonMode", "Adduct"])

def _clean_adduct(adduct: str) -> str:
    """
    Consistent encoding of adducts, including charge information.

    Parameters
    ----------
    adduct : str
        The original adduct string.

    Returns
    -------
    str
        The cleaned adduct string.
    """
    # Keep "]" for now to handle charge as "M+Ca]2"
    new_adduct = re.sub(r"[ ()\[]", "", adduct)
    # Find out whether the charge is specified at the end.
    charge, charge_sign = 0, None
    for i in reversed(range(len(new_adduct))):
        if new_adduct[i] in ("+", "-"):
            if charge_sign is None:
                charge, charge_sign = 1, new_adduct[i]
            else:
                # Keep increasing the charge for multiply charged ions.
                charge += 1
        elif new_adduct[i].isdigit():
            charge += int(new_adduct[i])
        else:
            # Only use charge if charge sign was detected;
            # otherwise no charge specified.
            if charge_sign is None:
                charge = 0
                # Special case to handle "M+Ca]2" -> missing sign, will remove
                # charge and try to calculate from parts later.
                if new_adduct[i] in ("]", "/"):
                    new_adduct = new_adduct[: i + 1]
            else:
                # Charge detected: remove from str.
                new_adduct = new_adduct[: i + 1]
            break
    # Now remove trailing delimiters after charge detection.
    new_adduct = re.sub("[\]/]", "", new_adduct)

    # Unknown adduct.
    if new_adduct.lower() in map(
        str.lower, ["?", "??", "???", "M", "M+?", "M-?", "unk", "unknown"]
    ):
        return "unknown"

    # Find neutral losses and additions.
    positive_parts, negative_parts = [], []
    for part in new_adduct.split("+"):
        pos_part, *neg_parts = part.split("-")
        positive_parts.append(_get_adduct_count(pos_part))
        for neg_part in neg_parts:
            negative_parts.append(_get_adduct_count(neg_part))
    mol = positive_parts[0]
    positive_parts = sorted(positive_parts[1:], key=operator.itemgetter(1))
    negative_parts = sorted(negative_parts, key=operator.itemgetter(1))
    # Handle weird Cat = [M]+ notation.
    if mol[1].lower() == "Cat".lower():
        mol = mol[0], "M"
        charge, charge_sign = 1, "+"

    # Calculate the charge from the individual components.
    if charge_sign is None:
        charge = sum(
            [
                count * config.charges.get(adduct, 0)
                for count, adduct in positive_parts
            ]
        ) + sum(
            [
                count * -abs(config.charges.get(adduct, 0))
                for count, adduct in negative_parts
            ]
        )
        charge_sign = "-" if charge < 0 else "+" if charge > 0 else ""

    cleaned_adduct = ["[", f"{mol[0] if mol[0] > 1 else ''}{mol[1]}"]
    if negative_parts:
        for count, adduct in negative_parts:
            cleaned_adduct.append(f"-{count if count > 1 else ''}{adduct}")
    if positive_parts:
        for count, adduct in positive_parts:
            cleaned_adduct.append(f"+{count if count > 1 else ''}{adduct}")
    cleaned_adduct.append("]")
    cleaned_adduct.append(
        f"{abs(charge) if abs(charge) > 1 else ''}{charge_sign}"
    )
    return "".join(cleaned_adduct)

def _get_adduct_count(adduct: str) -> Tuple[int, str]:
    """
    Split the adduct string in count and raw adduct.

    Parameters
    ----------
    adduct : str

    Returns
    -------
    Tuple[int, str]
        The count of the adduct and its raw value.
    """
    count, adduct = re.match(r"^(\d*)([A-Z]?.*)$", adduct).groups()
    count = int(count) if count else 1
    adduct = config.formulas.get(adduct, adduct)
    wrong_order = re.match(r"^([A-Z][a-z]*)(\d*)$", adduct)
    # Handle multimers: "M2" -> "2M".
    if wrong_order is not None:
        adduct, count_new = wrong_order.groups()
        count = int(count_new) if count_new else count
    return count, adduct

def _filter_pairs(pairs: pd.DataFrame, min_cosine: float) -> pd.DataFrame:
    """
    Only consider pairs with a cosine similarity that exceeds the given cosine
    threshold.

    Arguments
    ---------
    pairs : pd.DataFrame
        The tabular pairs retrieved from GNPS.
    min_cosine : float
        The minimum cosine used to retain high-quality pairs.

    Returns
    -------
    pd.DataFrame
        The pairs filtered by minimum cosine similarity.
    """
    return pairs[pairs["Cosine"] >= min_cosine]

def _filter_clusters(cluster_info: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster select as representative the scan with the highest
    precursor intensity.

    Arguments
    ---------
    cluster_info : pd.DataFrame
        The tabular cluster info retrieved from GNPS.

    Returns
    -------
    pd.DataFrame
        Clusters without duplicated spectra by keeping only the scan with the
        highest precursor intensity for each cluster.
    """
    cluster_info = (
        cluster_info.reindex(
            cluster_info.groupby("ClusterId")["PrecursorIntensity"].idxmax()
        )
        .dropna()
        .reset_index(drop=True)[
            ["ClusterId", "SuspectPrecursorMass", "SuspectUsi"]
        ]
    )
    return cluster_info

def _generate_suspects(
    ids: pd.DataFrame, pairs: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate suspects from identifications and aligned spectra pairs.
    Provenance about the spectra pairs is added from the summary.

    Arguments
    ---------
    ids : pd.DataFrame
        The filtered identifications.
    pairs : pd.DataFrame
        The filtered pairs.
    clusters : pd.DataFrame
        The filtered clustering information.

    Returns
    -------
    pd.DataFrame
        A DataFrame with information about both spectra forming a suspect
        identification.
    """
    # Form suspects of library and unidentified spectra pairs.
    suspects = pd.concat(
        [
            pd.merge(pairs, ids, left_on="ClusterId1", right_on="ClusterId")
            .drop(columns=["ClusterId", "ClusterId1"])
            .rename(columns={"ClusterId2": "ClusterId"}),
            pd.merge(pairs, ids, left_on="ClusterId2", right_on="ClusterId")
            .drop(columns=["ClusterId", "ClusterId2"])
            .rename(columns={"ClusterId1": "ClusterId"}),
        ],
        ignore_index=True,
        sort=False,
    )

    # Add provenance information for the library and suspect scans.
    suspects = pd.merge(suspects, clusters, on="ClusterId")
    suspects = suspects[
        [
            "InChI",
            "CompoundName",
            "Adduct",
            "IonSource",
            "Instrument",
            "IonMode",
            "Cosine",
            "LibraryUsi",
            "SuspectUsi",
            "LibraryPrecursorMass",
            "SuspectPrecursorMass",
        ]
    ]
    suspects["DeltaMass"] = (
        suspects["SuspectPrecursorMass"] - suspects["LibraryPrecursorMass"]
    )
    suspects["GroupDeltaMass"] = pd.Series(dtype=np.float32)
    suspects["AtomicDifference"] = pd.Series(dtype=str)
    suspects["Rationale"] = pd.Series(dtype=str)
    return suspects

def _load_unimod_tables_rows(unimod_file: str):
    """Parse UNIMOD *tables* XML (unimod_tables_1) and return rows with DeltaMass + title."""
    rows = []
    if not (unimod_file and os.path.exists(unimod_file)):
        logger.warning("UNIMOD XML not found at %s", unimod_file)
        return rows

    tree = ET.parse(unimod_file)
    root = tree.getroot()
    def local(tag): return tag.split('}', 1)[-1] if '}' in tag else tag

    count = 0
    for el in root.iter():
        if local(el.tag) != "modifications_row":
            continue
        attrs = el.attrib

        # Your probe showed these keys:
        mono = attrs.get("mono_mass")
        if mono is None:
            # fallback to average mass if you *really* want to (optional):
            mono = attrs.get("avge_mass")

        title = attrs.get("full_name") or attrs.get("code_name") or "UNIMOD modification"

        if mono is None:
            continue
        try:
            dm = float(mono)
        except ValueError:
            continue

        rows.append({
            "DeltaMass": dm,
            "AtomicDifference": [],          # (optional: derive from 'composition' if you want)
            "Rationale": f"UNIMOD: {title}",
            "Priority": 250,                 # keep lower than curated CSV
        })
        count += 1

    logger.info("Loaded %d modifications from UNIMOD tables XML %s", count, unimod_file)
    if count == 0:
        logger.warning("No usable modifications found in %s (looked at <modifications_row> attrs).", unimod_file)
    return rows

def _get_mass_shift_annotations(
    extra_annotations: Optional[str] = None,
    unimod_file: Optional[str] = None
) -> pd.DataFrame:
    cols = ["DeltaMass", "AtomicDifference", "Rationale", "Priority"]
    mass_shift_annotations = pd.DataFrame(columns=cols)

    # 1) Optional curated CSV
    if extra_annotations:
        try:
            df = (
                pd.read_csv(extra_annotations, usecols=["mz delta", "atomic difference", "rationale", "priority"])
                  .rename(columns={
                      "mz delta": "DeltaMass",
                      "atomic difference": "AtomicDifference",
                      "rationale": "Rationale",
                      "priority": "Priority",
                  })
            )
            df["AtomicDifference"] = df["AtomicDifference"].fillna("").apply(
                lambda s: [x.strip() for x in s.split(",")] if s else []
            )
            mass_shift_annotations = df
            logger.info("Loaded %d curated mass-shift annotations from CSV %s",
                        len(df), extra_annotations)
        except Exception as e:
            logger.warning("Error loading extra annotations %s: %s",
                           extra_annotations, e)

    # 2) UNIMOD from LOCAL XML (no network)
    rows = []
    unimod_rows = _load_unimod_tables_rows(unimod_file)
    if unimod_rows:
        mass_shift_annotations = pd.concat(
            [mass_shift_annotations, pd.DataFrame(unimod_rows)],
            ignore_index=True
        )
    else:
        logger.warning("No UNIMOD modifications parsed from %s", unimod_file)
    # 3) Add reverse entries
    ms = mass_shift_annotations.copy()
    ms["AtomicDifference"] = ms["AtomicDifference"].apply(
        lambda v: v if isinstance(v, list)
        else ([x.strip() for x in str(v).split(",")] if str(v).strip() else [])
    )
    ms_rev = ms.copy()
    ms_rev["DeltaMass"] = -ms_rev["DeltaMass"]
    ms_rev["Rationale"] = (ms_rev["Rationale"] + " (reverse)").str.replace(
        "unspecified (reverse)", "unspecified", regex=False
    )
    ms_rev["AtomicDifference"] = ms_rev["AtomicDifference"].apply(
        lambda row: [a[1:] if a.startswith("-") else f"-{a}" for a in row]
    )

    out = pd.concat([ms, ms_rev], ignore_index=True)
    out["AtomicDifference"] = out["AtomicDifference"].apply(lambda lst: ",".join(lst))
    for col, t in (("DeltaMass", np.float32), ("Priority", np.uint8)):
        if col in out:
            out[col] = out[col].astype(t)
    return out.sort_values("DeltaMass").reset_index(drop=True)

def _group_mass_shifts(
    suspects: pd.DataFrame,
    mass_shift_annotations: pd.DataFrame,
    interval_width: float,
    bin_width: float,
    peak_height: float,
    max_dist: float,
) -> pd.DataFrame:
    """
    Group similar mass shifts.

    Mass shifts are binned and the group delta mass is detected by finding
    peaks in the histogram. Grouped mass shifts are assigned potential
    explanations from the given mass shift annotations. If no annotation can be
    found for a certain group, the rationale and atomic difference will be
    marked as "unspecified". Ungrouped suspects will have a missing rationale
    and atomic difference.

    Arguments
    ---------
    suspects : pd.DataFrame
        The suspects from which mass shifts are grouped.
    mass_shift_annotations : pd.DataFrame
        Mass shift explanations.
    interval_width : float
        The size of the interval in which mass shifts are binned, centered
        around unit masses.
    bin_width : float
        The bin width used to construct the histogram.
    peak_height : float
        The minimum height for a peak to be considered as a group.
    max_dist : float
        The maximum mass difference that group members can have with the
        group's peak.

    Returns
    -------
    pd.DataFrame
        The suspects with grouped mass shifts and corresponding rationale (if
        applicable).
    """
    # Assign putative identifications to the mass shifts.
    for mz in np.arange(
        math.floor(suspects["DeltaMass"].min()),
        math.ceil(suspects["DeltaMass"].max() + interval_width),
        interval_width,
    ):
        suspects_interval = suspects[
            suspects["DeltaMass"].between(
                mz - interval_width / 2, mz + interval_width / 2
            )
        ]
        if len(suspects_interval) == 0:
            continue
        # Get peaks for frequent deltas in the histogram.
        bins = (
            np.linspace(
                mz - interval_width / 2,
                mz + interval_width / 2,
                int(interval_width / bin_width) + 1,
            )
            + bin_width / 2
        )
        hist, _ = np.histogram(suspects_interval["DeltaMass"], bins=bins)
        peaks_i, prominences = ssignal.find_peaks(
            hist,
            height=peak_height,
            distance=max_dist / bin_width,
            prominence=(None, None),
        )
        if len(peaks_i) == 0:
            continue
        # Assign deltas to their closest peak.
        mask_peaks = np.unique(
            np.hstack(
                [
                    suspects_interval.index[
                        suspects_interval["DeltaMass"].between(min_mz, max_mz)
                    ]
                    for min_mz, max_mz in zip(
                        bins[prominences["left_bases"]],
                        bins[prominences["right_bases"]],
                    )
                ]
            )
        )
        mz_diffs = np.vstack(
            [
                np.abs(suspects.loc[mask_peaks, "DeltaMass"] - peak)
                for peak in bins[peaks_i]
            ]
        )
        # Also make sure that delta assignments don't exceed the maximum
        # distance.
        # noinspection PyArgumentList
        mask_mz_diffs = mz_diffs.min(axis=0) < max_dist
        mz_diffs = mz_diffs[:, mask_mz_diffs]
        mask_peaks = mask_peaks[mask_mz_diffs]
        peak_assignments = mz_diffs.argmin(axis=0)
        # Assign putative explanations to the grouped mass shifts.
        for peak_i in range(len(peaks_i)):
            mask_delta_mz = mask_peaks[peak_assignments == peak_i]
            if len(mask_delta_mz) == 0:
                continue
            delta_mz = suspects.loc[mask_delta_mz, "DeltaMass"].mean()
            delta_mz_std = suspects.loc[mask_delta_mz, "DeltaMass"].std()
            if not np.isfinite(delta_mz_std) or delta_mz_std == 0:
                delta_mz_std = max_dist  # reasonable fallback
            suspects.loc[mask_delta_mz, "GroupDeltaMass"] = delta_mz
            putative_id = mass_shift_annotations[
                (mass_shift_annotations["DeltaMass"] - delta_mz).abs()
                < delta_mz_std
            ].sort_values(["Priority", "AtomicDifference", "Rationale"])
            if len(putative_id) == 0:
                for col in ("AtomicDifference", "Rationale"):
                    suspects.loc[mask_delta_mz, col] = "unspecified"
            else:
                for col in ("AtomicDifference", "Rationale"):
                    putative_id[col] = putative_id[col].fillna("unspecified")
                # Only use reverse explanations if no other explanations match.
                not_rev = ~putative_id["Rationale"].str.endswith("(reverse)")
                if not_rev.any():
                    putative_id = putative_id[not_rev]
                for col in ("AtomicDifference", "Rationale"):
                    suspects.loc[mask_delta_mz, col] = "|".join(
                        putative_id[col]
                    )

    suspects["DeltaMass"] = suspects["DeltaMass"].round(3)
    suspects["GroupDeltaMass"] = suspects["GroupDeltaMass"].round(3)
    return suspects.sort_values(
        ["CompoundName", "Adduct", "GroupDeltaMass"]
    ).reset_index(drop=True)[
        [
            "InChI",
            "CompoundName",
            "Adduct",
            "IonSource",
            "Instrument",
            "IonMode",
            "Cosine",
            "LibraryUsi",
            "SuspectUsi",
            "LibraryPrecursorMass",
            "SuspectPrecursorMass",
            "DeltaMass",
            "GroupDeltaMass",
            "AtomicDifference",
            "Rationale",
        ]
    ]

def _get_adduct_n_elements(adducts: pd.Series) -> pd.Series:
    """
    Determine how many components the adducts consist of.

    Parameters
    ----------
    adducts : pd.Series
        A Series with different adducts.

    Returns
    -------
    pd.Series
        A Series with the number of components for the corresponding adducts.
        Unknown adducts are assigned "infinity" components.
    """
    counts = []
    for adduct in adducts:
        if adduct == "unknown":
            counts.append(np.inf)
        else:
            n = sum(
                [
                    _get_adduct_count(split)[0]
                    for split in re.split(
                        r"[+-]",
                        adduct[adduct.find("[") + 1 : adduct.rfind("]")],
                    )
                ]
            )
            counts.append(n if n > 1 else np.inf)
    return pd.Series(counts)

if __name__ == "__main__":
    logging.basicConfig(
        format="{asctime} [{levelname}/{processName}] {message}",
        style="{",
        level=logging.INFO,
    )
    logging.captureWarnings(True)
    logger.setLevel(logging.INFO)

    generate_suspects()






# def _get_mass_shift_annotations(
#     extra_annotations: Optional[str] = None,
#     unimod_file: Optional[str] = None
# ) -> pd.DataFrame:
#     mass_shift_annotations = pd.DataFrame(
#         columns=["DeltaMass", "AtomicDifference", "Rationale", "Priority"]
#     )
#     if extra_annotations:
#         try:
#             mass_shift_annotations = pd.read_csv(
#                 extra_annotations,
#                 usecols=["mz delta", "atomic difference", "rationale", "priority"]
#             ).rename(
#                 columns={
#                     "mz delta": "DeltaMass",
#                     "atomic difference": "AtomicDifference",
#                     "rationale": "Rationale",
#                     "priority": "Priority"
#                 }
#             )
#             mass_shift_annotations["AtomicDifference"] = mass_shift_annotations["AtomicDifference"].str.split(",")
#         except Exception as e:
#             print(f"Error loading extra annotations {extra_annotations}: {e}")
#     # Add reversed modifications
#     mass_shift_annotations_rev = mass_shift_annotations.copy()
#     mass_shift_annotations_rev["DeltaMass"] *= -1
#     mass_shift_annotations_rev["Rationale"] = (
#         mass_shift_annotations_rev["Rationale"] + " (reverse)"
#     ).str.replace("unspecified (reverse)", "unspecified", regex=False)
#     mass_shift_annotations_rev["AtomicDifference"] = (
#         mass_shift_annotations_rev["AtomicDifference"]
#         .fillna("")
#         .apply(list)
#         .apply(lambda row: [a[1:] if a.startswith("-") else f"-{a}" for a in row])
#     )
#     mass_shift_annotations = pd.concat(
#         [mass_shift_annotations, mass_shift_annotations_rev],
#         ignore_index=True
#     )
#     mass_shift_annotations["AtomicDifference"] = mass_shift_annotations["AtomicDifference"].str.join(",")
#     for col, t in (("DeltaMass", np.float32), ("Priority", np.uint8)):
#         mass_shift_annotations[col] = mass_shift_annotations[col].astype(t)
#     return mass_shift_annotations.sort_values("DeltaMass").reset_index(drop=True)