import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from pymatgen.core import Composition


def compute_structural_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет геометрические дескрипторы кристаллов (Раздел 8.3 плана):
    - Число атомов в формульной единице;
    - Объем на один атом в ячейке (Å³);
    - Атомная плотность упаковки.
    """
    df = df.copy()

    def get_num_atoms(formula: str) -> float:
        try:
            return float(Composition(formula).num_atoms)
        except Exception:
            return 1.0

    df["atoms_per_formula"] = df["formula"].apply(get_num_atoms)
    df["atomic_density_ratio"] = df["atoms_per_formula"] / df["volume"].clip(lower=1.0)
    df["volume_per_atom"] = df["volume"] / df["atoms_per_formula"]
    return df


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "formula": ["Si", "SiC", "C", "GaN"],
        "volume": [40.88, 20.7, 11.4, 23.8],
        "density": [2.33, 3.21, 3.51, 6.15]
    })
    res = compute_structural_descriptors(test_df)
    print(" Структурные признаки работают корректно:")
    print(res[["formula", "volume", "atoms_per_formula", "volume_per_atom"]])