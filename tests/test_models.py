from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from src.screening.ranker import BetavoltaicLibrary


def test_saved_model_exists_and_loads():
    """Проверка: веса модели существуют и успешно загружаются."""
    model_path = Path("models/delta_eg_catboost.cbm")
    assert model_path.exists()
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    assert model.is_fitted()
    assert model.tree_count_ > 0


def test_database_queries():
    """Проверка: база SQLite отвечает на запросы и возвращает кандидатов."""
    lib = BetavoltaicLibrary()
    df = lib.get_top_candidates(isotope="Ni-63", top_k=5)
    assert len(df) == 5
    assert "formula" in df.columns
    assert "Eff_pct" in df.columns
    assert "Is_Immune" in df.columns


def test_find_benchmark_materials():
    """Проверка доступности эталонных полупроводников через API."""
    lib = BetavoltaicLibrary()
    for mat in ["Si", "C", "GaN", "SiC", "TiO2"]:
        res = lib.find_by_formula(mat)
        assert not res.empty, f"Эталон {mat} должен присутствовать в БД"
        row = res.iloc[0]
        assert row["band_gap_calibrated"] > 0
        assert row["theoretical_efficiency_pct"] > 0