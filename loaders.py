"""
Data loading utilities for all three APEX domains.
Handles DrugBank, ChEMBL, STRING, DisGeNET, CMIP6, ERA5,
Materials Project, and SuperCon datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("apex.data.loaders")


class DataLoader:
    """Unified data loader for APEX domains."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

    # ------------------------------------------------------------------
    # Drug discovery
    # ------------------------------------------------------------------
    def load_drugbank(self, path: str | None = None) -> pd.DataFrame:
        """Load DrugBank drug-target interaction data."""
        fpath = Path(path) if path else self.raw_dir / "drugbank_drug_target_interactions.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            logger.info("Loaded DrugBank: %d rows × %d cols", *df.shape)
            return df
        logger.info("DrugBank file not found — generating synthetic demo data")
        return self._generate_synthetic_drug_data()

    def load_chembl(self, path: str | None = None) -> pd.DataFrame:
        """Load ChEMBL bioactivity data."""
        fpath = Path(path) if path else self.raw_dir / "chembl_bioactivity.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            logger.info("Loaded ChEMBL: %d rows × %d cols", *df.shape)
            return df
        logger.info("ChEMBL file not found at %s", fpath)
        return pd.DataFrame()

    def load_string_ppi(self, path: str | None = None) -> pd.DataFrame:
        """Load STRING protein-protein interaction data."""
        fpath = Path(path) if path else self.raw_dir / "string_protein_interactions.tsv"
        if fpath.exists():
            df = pd.read_csv(fpath, sep="\t")
            logger.info("Loaded STRING PPI: %d rows × %d cols", *df.shape)
            return df
        logger.info("STRING file not found at %s", fpath)
        return pd.DataFrame()

    def load_disgenet(self, path: str | None = None) -> pd.DataFrame:
        """Load DisGeNET gene-disease associations."""
        fpath = Path(path) if path else self.raw_dir / "disgenet_gene_disease.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            logger.info("Loaded DisGeNET: %d rows × %d cols", *df.shape)
            return df
        logger.info("DisGeNET file not found at %s", fpath)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Climate
    # ------------------------------------------------------------------
    def load_climate_data(self, path: str | None = None) -> pd.DataFrame:
        """Load climate reanalysis data (CSV fallback for NetCDF)."""
        fpath = Path(path) if path else self.raw_dir / "climate_data.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            logger.info("Loaded climate data: %d rows × %d cols", *df.shape)
            return df
        logger.info("Climate data not found at %s", fpath)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    def load_materials_project(self, path: str | None = None) -> pd.DataFrame:
        """Load Materials Project data."""
        fpath = Path(path) if path else self.raw_dir / "materials_project.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            logger.info("Loaded Materials Project: %d rows × %d cols", *df.shape)
            return df
        logger.info("Materials Project data not found at %s", fpath)
        return pd.DataFrame()

    def load_supercon(self, path: str | None = None) -> pd.DataFrame:
        """Load SuperCon superconductor database."""
        fpath = Path(path) if path else self.raw_dir / "supercon_database.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            logger.info("Loaded SuperCon: %d rows × %d cols", *df.shape)
            return df
        logger.info("SuperCon data not found at %s", fpath)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Synthetic data generator (for demonstration / testing)
    # ------------------------------------------------------------------
    def _generate_synthetic_drug_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        Generate realistic synthetic drug-target interaction data
        for demonstration when real datasets are not yet placed.

        Feature distributions are calibrated to mimic DrugBank/ChEMBL
        molecular descriptor statistics.
        """
        rng = np.random.RandomState(42)

        molecular_weight = rng.lognormal(mean=5.6, sigma=0.4, size=n_samples)
        molecular_weight = np.clip(molecular_weight, 50, 1200)

        logP = rng.normal(loc=2.5, scale=1.8, size=n_samples)
        logP = np.clip(logP, -3, 10)

        hbd = rng.poisson(lam=1.5, size=n_samples)
        hbd = np.clip(hbd, 0, 10)

        hba = rng.poisson(lam=4.0, size=n_samples)
        hba = np.clip(hba, 0, 15)

        tpsa = rng.gamma(shape=3.0, scale=25.0, size=n_samples)
        tpsa = np.clip(tpsa, 0, 300)

        rotatable_bonds = rng.poisson(lam=4.0, size=n_samples)
        rotatable_bonds = np.clip(rotatable_bonds, 0, 25)

        # Binding affinity: causal relationships embedded
        binding_affinity = (
            -0.003 * molecular_weight
            + 0.4 * logP
            - 0.15 * hbd
            + 0.08 * hba
            - 0.005 * tpsa
            + 0.02 * rotatable_bonds
            + rng.normal(0, 0.3, n_samples)
        )
        binding_affinity = (binding_affinity - binding_affinity.min()) / (
            binding_affinity.max() - binding_affinity.min()
        )

        df = pd.DataFrame({
            "molecular_weight": np.round(molecular_weight, 2),
            "logP": np.round(logP, 3),
            "hbd": hbd.astype(int),
            "hba": hba.astype(int),
            "tpsa": np.round(tpsa, 2),
            "rotatable_bonds": rotatable_bonds.astype(int),
            "binding_affinity": np.round(binding_affinity, 4),
        })

        cache_path = self.processed_dir / "synthetic_drug_data.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Generated synthetic drug data: %d samples -> %s", n_samples, cache_path)

        return df

    # ------------------------------------------------------------------
    def load_domain_data(self, domain: str = "drug_discovery") -> pd.DataFrame:
        """Load the primary dataset for the configured domain."""
        if domain == "drug_discovery":
            return self.load_drugbank()
        elif domain == "climate":
            return self.load_climate_data()
        elif domain == "materials":
            return self.load_materials_project()
        else:
            raise ValueError(f"Unknown domain: {domain}")
