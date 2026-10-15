"""
Data ingestion and gene identifier annotation module for TCGA-COAD RNA-seq.
"""
import os
import pandas as pd
from typing import Optional


def load_tcga_expression(file_path: str) -> pd.DataFrame:
    """Loads TCGA FPKM expression data from TSV or compressed TSV.gz."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expression file not found: {file_path}")

    compression = 'gzip' if file_path.endswith('.gz') else None
    df = pd.read_csv(file_path, sep='\t', index_col=0, compression=compression)
    return df


def convert_ensembl_to_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes gene identifiers from Ensembl IDs (ENSG*) to HGNC symbols.
    Extracts embedded symbols if present, or queries MyGeneInfo API.
    """
    first_id = str(df.index[0])
    if not first_id.startswith('ENSG'):
        return df

    # Method 1: Embedded symbol in UCSC Xena format (e.g., ENSG00000006625.18|SCARA5)
    if '|' in first_id:
        df.index = [idx.split('|')[1] if '|' in idx else idx for idx in df.index]
        return df

    # Method 2: Query MyGeneInfo API
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        ensembl_ids = [str(idx).split('.')[0] for idx in df.index]
        results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')

        id_to_symbol = {r['query']: r['symbol'] for r in results if 'symbol' in r}
        new_index = [id_to_symbol.get(str(idx).split('.')[0], idx) for idx in df.index]
        df.index = new_index
    except ImportError:
        print("[Warning] 'mygene' not installed; continuing with original Ensembl IDs.")
    except Exception as e:
        print(f"[Warning] Failed to fetch symbols via MyGeneInfo: {e}")

    return df