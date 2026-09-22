import sys
from pathlib import Path

# Гарантируем, что Python видит корень проекта при любом способе запуска
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import sqlite3
import numpy as np
import pandas as pd
from src.database.schema import CREATE_TABLES_SQL
from src.physics.betavoltaics import enrich_with_betavoltaic_metrics

def calculate_radiation_displacement_energy(
    melting_temp_k: float,
    band_gap_ev: float,
    formation_energy_ev: float = 0.0
) -> float:
    """
    Полуэмпирическая прокси-оценка пороговой энергии смещения атомов Ed (в эВ).
    Основана на модели Кинчина-Пиза: Ed строго пропорциональна
    температуре плавления решетки и силе химических связей (Eg).
    """
    if pd.isna(melting_temp_k) or melting_temp_k <= 0:
        melting_temp_k = 1200.0
        
    ed = 0.0075 * melting_temp_k * (1.0 + 0.12 * band_gap_ev) + 0.5 * abs(formation_energy_ev)
    return float(np.clip(ed, 10.0, 100.0))


def build_database(
    calibrated_path: str = "data/03_features/materials_calibrated.parquet",
    db_path: str = "data/05_database/betavoltaic_library.db"
):
    print(" Формирование итоговой реляционной базы данных...")
    
    if not os.path.exists(calibrated_path):
        raise FileNotFoundError(f"Файл {calibrated_path} не найден! Запусти delta_learner.py.")

    df = pd.read_parquet(calibrated_path)
    print(f"  Загружено материалов: {len(df)}")

    # 1. Расчет прокси Ed
    print("  [1/4] Расчет полуэмпирической энергии смещения Ed и индекса стойкости...")
    melting_col = "MagpieData mean MeltingT" if "MagpieData mean MeltingT" in df.columns else None
    
    if melting_col:
        df["ed_est_ev"] = [
            calculate_radiation_displacement_energy(tm, eg, fe)
            for tm, eg, fe in zip(df[melting_col], df["band_gap_calibrated"], df["formation_energy"])
        ]
    else:
        df["ed_est_ev"] = [
            calculate_radiation_displacement_energy(1500.0, eg, fe)
            for eg, fe in zip(df["band_gap_calibrated"], df["formation_energy"])
        ]

    min_ed, max_ed = df["ed_est_ev"].min(), df["ed_est_ev"].max()
    df["radiation_resistance_score"] = ((df["ed_est_ev"] - min_ed) / (max_ed - min_ed)) * 100.0

    # 2. Физический расчет бетавольтаики для 4 изотопов на КАЛИБРОВАННОЙ зоне
    print("  [2/4] Физическое моделирование бетавольтаики для 4 изотопов (Ni-63, H-3, C-14, Pm-147)...")
    df = enrich_with_betavoltaic_metrics(df, band_gap_col="band_gap_calibrated")

    # 3. Инициализация SQLite
    print("  [3/4] Создание реляционных таблиц SQLite...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(CREATE_TABLES_SQL)

    # 4. Запись таблиц
    print("  [4/4] Запись данных в БД...")
    materials_table = df[[
        "mp_id", "formula", "crystal_system", "density", "volume", 
        "e_above_hull", "formation_energy"
    ]]
    materials_table.to_sql("materials", conn, if_exists="replace", index=False)

    electronic_table = df[[
        "mp_id", "band_gap_dft", "delta_eg_predicted", "band_gap_calibrated",
        "eps_ehp_ev", "Voc_est_v", "theoretical_efficiency_pct"
    ]]
    electronic_table.to_sql("electronic_properties", conn, if_exists="replace", index=False)

    perf_cols = [
        "mp_id", "ed_est_ev", "radiation_resistance_score",
        "carriers_per_electron_Ni63", "penetration_depth_um_Ni63",
        "carriers_per_electron_H3", "penetration_depth_um_H3",
        "carriers_per_electron_C14", "penetration_depth_um_C14",
        "carriers_per_electron_Pm147", "penetration_depth_um_Pm147"
    ]
    df[perf_cols].to_sql("betavoltaic_performance", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

    print(f"\n Реляционная база данных успешно сохранена в: {db_path}")
    print(f" Записано записей: {len(df)} в 3 связанные таблицы!")

if __name__ == "__main__":
    build_database()