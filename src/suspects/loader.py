from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import re, os
import glob
import joblib
import logging
import tqdm
from typing import Iterable

logger = logging.getLogger("suspect_library")

# This contains functions that were copied from suspects.py and changed from being helpers to standalone functions (just for clarity)
def _pick_one(folder: Path, pattern: str) -> Path:
    """
    Replaces the old way of picking the first file matching a pattern in a folder. 
    Raises an error if no file is found.
    """
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return files[0]

def _get_cluster(base: Path, msv_id: str) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]
]:
    """
    Parse cluster information for the living data analysis with the given
    MassIVE identifier.

    Parameters
    ----------
    data_dir : str
        The directory of the living data.
    msv_id : str
        The MassIVE identifier of the dataset in the living data analysis.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of the identifications, pairs, and cluster_info DataFrames.
    """
    ids_path   = base / "IDENTIFICATIONS" / f"{msv_id}_identifications.tsv"
    pairs_path = base / "PAIRS"           / f"{msv_id}_pairs.tsv"
    clust_path = base / "CLUSTERINFO"     / f"{msv_id}_clustering.tsv"

    if not (ids_path.exists() and pairs_path.exists() and clust_path.exists()):
        return None, None, None

    ids = pd.read_csv(
        ids_path, sep="\t",
        usecols=["Compound_Name","Ion_Source","Instrument","IonMode","Adduct",
                 "Precursor_MZ","INCHI","SpectrumID","#Scan#","MZErrorPPM","SharedPeaks"],
        dtype={"Precursor_MZ": np.float32,"#Scan#": "string","MZErrorPPM": np.float32,"SharedPeaks": np.uint8},
    ).rename(columns={
        "Compound_Name": "CompoundName",
        "Ion_Source": "IonSource",
        "Precursor_MZ": "LibraryPrecursorMass",
        "INCHI": "InChI",
        "SpectrumID": "LibraryUsi",
        "#Scan#": "ClusterId",
        "MZErrorPPM": "MzErrorPpm",
    })
    ids["ClusterId"] = f"{msv_id}:scan:" + ids["ClusterId"].astype(str)
    ids["LibraryUsi"] = "mzspec:GNPS:GNPS-LIBRARY:accession:" + ids["LibraryUsi"].astype(str)

    # Pairs
    pairs = pd.read_csv(
        pairs_path, sep="\t",
        usecols=["CLUSTERID1","CLUSTERID2","Cosine"],
        dtype={"CLUSTERID1": "string","CLUSTERID2": "string","Cosine": np.float32},
    ).rename(columns={"CLUSTERID1":"ClusterId1","CLUSTERID2":"ClusterId2"})
    pairs["ClusterId1"] = f"{msv_id}:scan:" + pairs["ClusterId1"].astype(str)
    pairs["ClusterId2"] = f"{msv_id}:scan:" + pairs["ClusterId2"].astype(str)

    # Clusters
    clust = pd.read_csv(
        clust_path, sep="\t",
        usecols=["cluster index","sum(precursor intensity)","parent mass","Original_Path","ScanNumber"],
        dtype={"cluster index": "string","sum(precursor intensity)": np.float32,"parent mass": np.float32,"ScanNumber": "string"},
    ).dropna(subset=["ScanNumber"])
    clust = clust[clust["ScanNumber"].astype("Int64").fillna(0) >= 0].copy()
    clust = clust.rename(columns={
        "cluster index": "ClusterId",
        "sum(precursor intensity)": "PrecursorIntensity",
        "parent mass": "SuspectPrecursorMass",
    })
    clust["ClusterId"] = f"{msv_id}:scan:" + clust["ClusterId"].astype(str)
    clust["SuspectUsi"] = (
        "mzspec:" + msv_id + ":" +
        clust["Original_Path"].apply(os.path.basename) + ":scan:" +
        clust["ScanNumber"]
    )
    clust = clust.drop(columns=["Original_Path","ScanNumber"])

    return ids, pairs, clust

def generate_suspects_global(global_network_dir: str, task_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transform cluster information for the global molecular network built on top
    of the living data analysis.

    Parameters
    ----------
    global_network_dir : str
        The directory with the output of the molecular networking job.
    task_id : str
        The GNPS task identifier of the molecular networking job.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of the identifications, pairs, and cluster_info DataFrames.
    """
    folder = Path(global_network_dir)
    filename_specnets = _pick_one(folder / "result_specnets_DB", "*.*")
    filename_networkedges = _pick_one(folder / "networkedges", "*.*")
    filename_clusterinfosummary = _pick_one(folder / "clusterinfosummary", "*.*")
    ids = pd.read_csv(
        filename_specnets,
        sep="\t",
        usecols=[
            "Compound_Name",
            "Ion_Source",
            "Instrument",
            "IonMode",
            "Adduct",
            "Precursor_MZ",
            "INCHI",
            "SpectrumID",
            "#Scan#",
            "MZErrorPPM",
            "SharedPeaks",
        ],
        dtype={
            "Precursor_MZ": np.float32,
            "#Scan#": str,
            "MzErrorPpm": np.float32,
            "SharedPeaks": np.uint8,
        },
    )
    ids = ids.rename(
        columns={
            "Compound_Name": "CompoundName",
            "Ion_Source": "IonSource",
            "Precursor_MZ": "LibraryPrecursorMass",
            "INCHI": "InChI",
            "SpectrumID": "LibraryUsi",
            "#Scan#": "ClusterId",
            "MZErrorPPM": "MzErrorPpm",
        }
    )
    ids["ClusterId"] = "GLOBAL_NETWORK:scan:" + ids["ClusterId"]
    ids["LibraryUsi"] = (
        "mzspec:GNPS:GNPS-LIBRARY:accession:" + ids["LibraryUsi"]
    )
    
    pairs = pd.read_csv(
        filename_networkedges,
        sep="\t",
        header=0,
        names=["ClusterId1", "ClusterId2", "Cosine"],
        usecols=[0, 1, 4],
        dtype={"ClusterId1": str, "ClusterId2": str, "Cosine": np.float32},
    )
    for col in ("ClusterId1", "ClusterId2"):
        pairs[col] = "GLOBAL_NETWORK:scan:" + pairs[col]
    clust = pd.read_csv(
        filename_clusterinfosummary,
        sep="\t",
        usecols=[
            "cluster index",
            "sum(precursor intensity)",
            "parent mass",
            "number of spectra",
        ],
        dtype={
            "cluster index": str,
            "sum(precursor intensity)": np.float32,
            "parent mass": np.float32,
            "number of spectra": np.uint32,
        },
    )
    clust = clust.rename(
        columns={
            "cluster index": "ClusterId",
            "sum(precursor intensity)": "PrecursorIntensity",
            "parent mass": "SuspectPrecursorMass",
        }
    )
    # Fix because cleaning on GNPS up didn't work.
    clust = clust[clust["number of spectra"] >= 3]
    suspect_usi = f"mzspec:GNPS:TASK-{task_id}-spectra/specs_ms.mgf:scan:"
    clust["SuspectUsi"] = suspect_usi + clust["ClusterId"]
    clust["ClusterId"] = "GLOBAL_NETWORK:scan:" + clust["ClusterId"]
    clust = clust.drop(columns="number of spectra")
    return ids, pairs, clust

REQUIRED_COLS: Iterable[str] = [
    "#Scan#",
    "Compound_Name",
    "Ion_Source",
    "Instrument",
    "Adduct",
    "LibMZ",
    "INCHI",
    "IonMode",
    "MZErrorPPM",
    "SharedPeaks",
    "SpectrumFile",
    "SpectrumID",
]

RENAME_MAP = {
    "Compound_Name": "CompoundName",
    "Ion_Source": "IonSource",
    "LibMZ": "LibraryPrecursorMass",
    "INCHI": "InChI",
    "MZErrorPPM": "MzErrorPpm",
}

DTYPE_MAP = {
    "#Scan#": "string",
    "LibMZ": np.float32,
    "MZErrorPPM": np.float32,
    "SharedPeaks": np.uint8,
}

def read_ids(ids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a GNPS library searching results DataFrame into the identifications DataFrame.

    Parameters
    ----------
    ids_df : pd.DataFrame
        DataFrame with the same columns as the GNPS library search TSV.

    Returns
    -------
    pd.DataFrame
        The identifications DataFrame.
    """
    print(ids_df.columns)
    missing = [c for c in REQUIRED_COLS if c not in ids_df.columns]
    if missing:
        raise ValueError(f"read_ids: missing required columns: {missing}")

    ids = ids_df.loc[:, list(REQUIRED_COLS)].copy()

    for col, dt in DTYPE_MAP.items():
        if col in ids.columns:
            ids[col] = ids[col].astype(dt)

    ids = ids.rename(columns=RENAME_MAP)

    ids["LibraryUsi"] = "mzspec:GNPS:GNPS-LIBRARY:accession:" + ids["SpectrumID"].astype("string")
    ids["ClusterId"] = (
        ids["SpectrumFile"].apply(lambda fn: os.path.splitext(str(fn))[0])
        + ":scan:"
        + ids["#Scan#"].astype("string")
    )


def load_living_data(living_data_base_dir: str, n_jobs: int = -1) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    base = Path(living_data_base_dir)
    ids_dir, pairs_dir, clust_dir = base/"IDENTIFICATIONS", base/"PAIRS", base/"CLUSTERINFO"
    if not (ids_dir.exists() and pairs_dir.exists() and clust_dir.exists()):
        return [], [], []

    # discover MSV ids like the original
    msv_ids = []
    for fn in (clust_dir.glob("*.tsv")):
        m = re.search(r"(MSV\d{9})_clustering", fn.name)
        if m:
            msv_ids.append(m.group(1))

    ids_list, pairs_list, clusters_list = [], [], []
    for res in joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_get_cluster)(base, msv) for msv in tqdm.tqdm(msv_ids, desc="Datasets processed", unit="dataset")
    ):
        if res is None:
            continue
        i, p, c = res
        if i is not None and p is not None and c is not None:
            ids_list.append(i); pairs_list.append(p); clusters_list.append(c)

    return ids_list, pairs_list, clusters_list