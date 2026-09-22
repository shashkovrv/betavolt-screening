from pathlib import Path
from catboost import CatBoostRegressor
from src.screening.ranker import BetavoltaicLibrary


def test_saved_model_exists_and_loads():
    """Проверка: веса модели существуют и успешно загружаются."""
    model_path = Path("models/delta_eg_catboost.cbm")
    assert model_path.exists()
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    assert model.is_fitted()


def test_database_queries():
    """Проверка: база SQLite отвечает на запросы и возвращает кандидатов."""
    lib = BetavoltaicLibrary()
    df = lib.get_top_candidates(isotope="Ni-63", top_k=3)
    assert len(df) == 3
    assert "formula" in df.columns
    assert "Eff_pct" in df.columns