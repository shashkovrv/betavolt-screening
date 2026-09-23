import sys
from pathlib import Path

# Гарантируем, что Python видит корень проекта при любом способе запуска
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import re
import sqlite3
import numpy as np
import pandas as pd
from src.database.schema import CREATE_TABLES_SQL
from src.physics.betavoltaics import enrich_with_betavoltaic_metrics


def classify_material_and_viability(formula: str) -> tuple[str, int]:
    """
    Классифицирует кристаллическое соединение по химическому классу
    и определяет его технологическую жизнеспособность для бетавольтаики.
    
    Исключает:
    - Растворимые соли и галогениды (F, Cl, Br, I);
    - Гидриды и гидроксиды (H);
    - Токсичные цианиды (CN) и оксоанионы (CO3, NO3, SO4, PO4);
    - Нестабильные ионные ацетилиды щелочных/щелочноземельных металлов (CaC2, SrC2, LiC);
    - Радиоактивные актиниды и благородные газы.
    """
    f = str(formula).strip()

    # 1. Проверка на нежизнеспособные / нестабильные классы
    if any(h in f for h in ["F", "Cl", "Br", "I"]):
        return "Исключен (Галогенид/Соль)", 0
    if "H" in f:
        return "Исключен (Гидрид/Гидроксид)", 0
    if any(grp in f for grp in ["CN", "CO3", "NO3", "SO4", "PO4", "NH4"]):
        return "Исключен (Сложная соль/Цианид)", 0
    if re.match(r"^(Li|Na|K|Rb|Cs|Ca|Sr|Ba)[0-9]*C[0-9]*$", f):
        return "Исключен (Ионный ацетилид)", 0
    if any(act in f for act in ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "He", "Ne", "Ar", "Kr", "Xe"]):
        return "Исключен (Радиоактивный/Инертный)", 0

    elems = set(re.findall(r"[A-Z][a-z]?", f))

    # 2. Ковалентные полупроводники (IV, III-V, карбиды, бориды, нитриды, фосфиды)
    if ("N" in elems or "C" in elems or "B" in elems or "P" in elems or "Si" in elems or "As" in elems) and \
       ("O" not in elems) and ("S" not in elems) and ("Se" not in elems) and ("Te" not in elems):
        return "Ковалентные (IV, III-V, карбиды, бориды, нитриды)", 1

    # 3. Оксидные полупроводники
    if "O" in elems:
        if len(elems) <= 3:
            return "Оксидные полупроводники (простые и тройные)", 1
        else:
            return "Сложные оксиды", 1

    # 4. Халькогениды (II-VI, слоистые TMDC)
    if any(ch in elems for ch in ["S", "Se", "Te"]):
        if len(elems) <= 3:
            return "Халькогениды (II-VI, дихалькогениды)", 1
        else:
            return "Сложные халькогениды", 1

    return "Прочие полупроводники", 1


def calculate_radiation_displacement_energy(
    melting_temp_k: float,
    band_gap_ev: float,
    formation_energy_ev: float = 0.0
) -> float:
    """
    Полуэмпирическая прокси-оценка пороговой энергии смещения атомов Ed (в эВ).
    Основана на расширенной модели Кинчина-Пиза: Ed пропорциональна
    ширине запрещенной зоны Eg (энергии ковалентного расщепления),
    энтальпии образования соединения |ΔHf| и максимальной температуре
    плавления элементов в кристаллической решетке.
    """
    fe_abs = abs(formation_energy_ev) if pd.notna(formation_energy_ev) else 0.0
    tm = melting_temp_k if (pd.notna(melting_temp_k) and melting_temp_k > 0) else 1200.0
    
    # Физическая прокси-модель Ed:
    ed = 10.0 + 2.2 * band_gap_ev + 3.0 * fe_abs + 0.0025 * tm
    return float(np.clip(ed, 10.0, 50.0))


def build_database(
    calibrated_path: str = "data/03_features/materials_calibrated.parquet",
    db_path: str = "data/05_database/betavoltaic_library.db"
):
    print(" Формирование итоговой реляционной базы данных...")
    
    if not os.path.exists(calibrated_path):
        raise FileNotFoundError(f"Файл {calibrated_path} не найден! Запусти delta_learner.py.")

    df = pd.read_parquet(calibrated_path)
    print(f"  Загружено материалов: {len(df)}")

    # 1. Химическая классификация и жизнеспособность
    print("  [1/5] Химическая классификация и оценка жизнеспособности...")
    classified = [classify_material_and_viability(f) for f in df["formula"]]
    df["material_class"] = [c[0] for c in classified]
    df["is_viable"] = [c[1] for c in classified]

    # 2. Расчет полуэмпирической энергии смещения Ed и индекса стойкости
    print("  [2/5] Расчет физической энергии смещения Ed и индекса стойкости...")
    tm_col = "MagpieData maximum MeltingT" if "MagpieData maximum MeltingT" in df.columns else "MagpieData mean MeltingT"
    
    if tm_col in df.columns:
        df["ed_est_ev"] = [
            calculate_radiation_displacement_energy(tm, eg, fe)
            for tm, eg, fe in zip(df[tm_col], df["band_gap_calibrated"], df["formation_energy"])
        ]
    else:
        df["ed_est_ev"] = [
            calculate_radiation_displacement_energy(1500.0, eg, fe)
            for eg, fe in zip(df["band_gap_calibrated"], df["formation_energy"])
        ]

    min_ed, max_ed = df["ed_est_ev"].min(), df["ed_est_ev"].max()
    df["radiation_resistance_score"] = ((df["ed_est_ev"] - min_ed) / (max_ed - min_ed)) * 100.0

    # 3. Физический расчет бетавольтаики для 4 изотопов на КАЛИБРОВАННОЙ зоне
    print("  [3/5] Физическое моделирование бетавольтаики для 4 изотопов (Ni-63, H-3, C-14, Pm-147)...")
    df = enrich_with_betavoltaic_metrics(df, band_gap_col="band_gap_calibrated")

    # 4. Инициализация SQLite
    print("  [4/5] Создание реляционных таблиц SQLite...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(CREATE_TABLES_SQL)

    # 5. Запись таблиц
    print("  [5/5] Запись данных в БД...")
    materials_table = df[[
        "mp_id", "formula", "crystal_system", "density", "volume", 
        "e_above_hull", "formation_energy", "material_class", "is_viable"
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
    print(f" Жизнеспособных полупроводников: {df['is_viable'].sum()} из {len(df)}")


if __name__ == "__main__":
    build_database()