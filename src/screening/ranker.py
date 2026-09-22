import sys
from pathlib import Path

# Гарантируем корректный импорт при запуске кнопкой Play в VS Code
sys.path.append(str(Path(__file__).resolve().parents[2]))

import sqlite3
import pandas as pd


class BetavoltaicLibrary:
    """
    Программный интерфейс (Python API) к библиотеке бетавольтаических материалов.
    Реализует требования раздела 8.4.5 дипломного плана.
    """
    def __init__(self, db_path: str = "data/05_database/betavoltaic_library.db"):
        self.db_path = db_path
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"База данных {self.db_path} не найдена! Сначала запусти repository.py.")

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def get_full_dataframe(self) -> pd.DataFrame:
        """Возвращает полную объединенную таблицу всех 22 263 материалов со всеми физическими свойствами."""
        sql = """
        SELECT 
            m.mp_id, m.formula, m.crystal_system, m.density, m.volume, m.e_above_hull,
            e.band_gap_dft, e.delta_eg_predicted, e.band_gap_calibrated, e.eps_ehp_ev, e.Voc_est_v, e.theoretical_efficiency_pct,
            p.ed_est_ev, p.radiation_resistance_score,
            p.carriers_per_electron_Ni63, p.penetration_depth_um_Ni63,
            p.carriers_per_electron_H3, p.penetration_depth_um_H3,
            p.carriers_per_electron_C14, p.penetration_depth_um_C14,
            p.carriers_per_electron_Pm147, p.penetration_depth_um_Pm147
        FROM materials m
        JOIN electronic_properties e ON m.mp_id = e.mp_id
        JOIN betavoltaic_performance p ON m.mp_id = p.mp_id
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(sql, conn)

    def find_by_formula(self, formula: str) -> pd.DataFrame:
        """Поиск конкретного полупроводника по формуле (например: 'Si', 'C', 'GaN', 'TiO2')."""
        sql = """
        SELECT 
            m.formula, m.mp_id, m.crystal_system, m.density,
            e.band_gap_dft, e.band_gap_calibrated, e.theoretical_efficiency_pct,
            p.ed_est_ev, p.radiation_resistance_score,
            p.penetration_depth_um_Ni63, p.penetration_depth_um_H3
        FROM materials m
        JOIN electronic_properties e ON m.mp_id = e.mp_id
        JOIN betavoltaic_performance p ON m.mp_id = p.mp_id
        WHERE m.formula = ?
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(sql, conn, params=(formula,))

    def get_top_candidates(
        self, 
        isotope: str = "Ni-63", 
        min_efficiency: float = 18.0, 
        min_radiation_score: float = 25.0,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Многокритериальный отбор лучших стабильных материалов:
        - Термодинамически стабильные (e_above_hull <= 0.01 эВ/атом)
        - Высокий КПД
        - Высокая радиационная стойкость
        """
        clean_tag = isotope.replace("-", "")
        sql = f"""
        SELECT 
            m.formula, m.mp_id, m.crystal_system, m.density,
            ROUND(e.band_gap_calibrated, 2) as Eg_calib_eV, 
            ROUND(e.theoretical_efficiency_pct, 2) as Eff_pct,
            ROUND(p.ed_est_ev, 1) as Ed_eV, 
            ROUND(p.radiation_resistance_score, 1) as Rad_Score,
            ROUND(p.penetration_depth_um_{clean_tag}, 2) as Depth_um,
            CAST(p.carriers_per_electron_{clean_tag} AS INTEGER) as Pairs_per_e
        FROM materials m
        JOIN electronic_properties e ON m.mp_id = e.mp_id
        JOIN betavoltaic_performance p ON m.mp_id = p.mp_id
        WHERE m.e_above_hull <= 0.01 
          AND e.theoretical_efficiency_pct >= {min_efficiency}
          AND p.radiation_resistance_score >= {min_radiation_score}
        ORDER BY (e.theoretical_efficiency_pct * 0.6 + p.radiation_resistance_score * 0.4) DESC
        LIMIT {top_k}
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(sql, conn)


if __name__ == "__main__":
    print("=================================================================")
    print("  ТЕСТИРОВАНИЕ PYTHON API БИБЛИОТЕКИ (BetavoltaicLibrary)")
    print("=================================================================\n")
    
    lib = BetavoltaicLibrary()

    # 1. Проверяем эталоны
    print("1. Поиск свойств эталонного Кремния (Si):")
    si_df = lib.find_by_formula("Si")
    print(si_df.to_string(index=False))

    print("\n2. Поиск свойств эталонного Алмаза (C):")
    c_df = lib.find_by_formula("C")
    print(c_df.to_string(index=False))

    # 2. Ищем Топ-5 перспективных материалов под Никель-63
    print("\n=================================================================")
    print("  ТОП-5 ПЕРСПЕКТИВНЫХ МАТЕРИАЛОВ ДЛЯ БАТАРЕЕК НА НИКЕЛЕ-63:")
    print("=================================================================")
    top_ni = lib.get_top_candidates(isotope="Ni-63", min_efficiency=19.0, top_k=5)
    print(top_ni.to_string(index=False))